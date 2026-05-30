"""Staged tracking pipeline orchestration.

The heavy helper logic lives in:
- inference_utils.py: atomic writes, parquet chunks, normalization, tracker checkpoints, frame processing
- image_utils.py: bbox/crop/render adapters

This file should mostly contain stage classes and their run order.
"""
from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from boxing_project.tracking.global_clustering import GlobalTrackClusterer
from boxing_project.tracking.image_utils import (
    build_visual_detections_from_rows,
    keypoints_to_intersection_bbox,
    render_tracking_overlays,
)
from boxing_project.tracking.inference_utils import (
    FRAMES_METADATA_COLUMNS,
    GLOBAL_MAP_COLUMNS,
    LOCAL_TRACKS_COLUMNS,
    OPENPOSE_RESULTS_COLUMNS,
    PREPARED_DETECTIONS_COLUMNS,
    TRACKING_LOGS_COLUMNS,
    TRACK_STATES_COLUMNS,
    atomic_write_json,
    atomic_write_parquet,
    chunk_path,
    collect_tracker_state_rows,
    detections_to_prepared_rows,
    df_by_frame,
    first_by_frame,
    free,
    iter_batches,
    load_manifest,
    load_tracker_checkpoint,
    normalize_keypoints_xy,
    normalize_kp_conf,
    prepared_rows_to_detections,
    preprocess_image,
    prepare_frame_detections_from_keypoints,
    read_chunked_parquet_for_frames,
    save_tracker_checkpoint,
    update_tracker_from_detections,
)

from boxing_project.tracking.progress_utils import RichStageProgress


@dataclass
class PipelineContext:
    opWrapper: Any
    tracker: Any
    app_embedder: Any
    sb_cfg: dict
    images: list[Path]
    save_width: int
    save_dir: Path
    save_log: bool
    restore_mode: bool
    cfg: dict
    graph_clustering_params: dict
    select_top_with_nearest: Any
    extract_features_with_hsv: Any
    build_fused_appearance_embedding_with_mask: Any


class BaseStage:
    name = "base"
    progress_key = "stage"
    progress_label = "Stage"
    progress_step = "[?/?]"

    def __init__(self, ctx: PipelineContext, progress: RichStageProgress):
        self.ctx = ctx
        self.progress = progress
        self.stage_dir = self.ctx.save_dir / self.name
        self.manifest_path = self.stage_dir / "manifest.json"

    def prepare_output(self) -> None:
        if not self.ctx.restore_mode and self.stage_dir.exists():
            shutil.rmtree(self.stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def manifest(self) -> dict[str, Any]:
        return load_manifest(self.manifest_path)

    def write_manifest(self, status: str, **extra) -> None:
        atomic_write_json(self.manifest_path, {"stage": self.name, "status": status, **extra})

    def start_progress(self, total: int, suffix: str = "") -> None:
        desc = f"{self.progress_step} {self.progress_label}{suffix}"
        self.progress.add(self.progress_key, desc, total=max(int(total), 1))

    def progress_update(self, *, frame=None, total=None, batch=0, op=None, advance=None, completed=None, force=False) -> None:
        parts = [f"{self.progress_step} {self.progress_label}"]
        if op:
            parts.append(str(op))
        if frame is not None and total is not None:
            parts.append(f"frame {frame}/{total}")
        parts.append(f"batch {batch}")
        kwargs = {"description": " | ".join(parts)}
        if advance is not None:
            kwargs["advance"] = advance
        if completed is not None:
            kwargs["completed"] = completed
        if force:
            kwargs["force"] = True
        self.progress.update(self.progress_key, **kwargs)

    def restore_completed(self, total: int, *, needed_paths: Iterable[Path] = (), needed_globs: Iterable[tuple[Path, str]] = ()) -> bool:
        if not self.ctx.restore_mode or self.manifest().get("status") != "completed":
            return False
        if any(not p.exists() for p in needed_paths):
            return False
        if any(not any(d.glob(pattern)) for d, pattern in needed_globs):
            return False
        self.start_progress(total, " | op=RESTORE | dev=IO")
        self.progress_update(completed=max(total, 1), frame=total, total=total, op="restored complete", force=True)
        self.progress.message(f"{self.progress_label} restore: already completed {total}/{total}")
        return True

    def cfg_int(self, section: str, key: str, default: int) -> int:
        return int(self.ctx.cfg.get(section, {}).get(key, default))

    def require(self, path: Path, what: str) -> None:
        if not path.exists():
            raise RuntimeError(f"Missing {what}: {path}")


class PreprocessingStage(BaseStage):
    name, progress_key, progress_label, progress_step = "preprocessed", "pre", "Preprocessing", "[1/6]"

    def run(self) -> None:
        self.prepare_output()
        from boxing_project.shot_boundary.inference import ShotBoundaryInferConfig, ShotBoundaryInferencer

        frames_dir = self.stage_dir / "frames"
        det_chunks_dir = self.stage_dir / "openpose_chunks"
        frames_dir.mkdir(parents=True, exist_ok=True)
        det_chunks_dir.mkdir(parents=True, exist_ok=True)

        total = len(self.ctx.images)
        out_meta = self.stage_dir / "frames_metadata.parquet"
        if self.restore_completed(total, needed_paths=[out_meta], needed_globs=[(det_chunks_dir, "openpose_*.parquet")]):
            return

        cfg = self.ctx.cfg.get("preprocessing", {})
        batch_size = int(cfg.get("batch_size", 32))
        flush_every = max(1, int(cfg.get("checkpoint_every_frames", batch_size)))
        manifest = self.manifest()
        keep_until = max(0, min(int(manifest.get("last_completed_frame", 0) or 0), total)) if self.ctx.restore_mode else 0
        start = keep_until + 1

        if self.ctx.restore_mode and total and keep_until >= total:
            raise RuntimeError(
                "Restore requested, but preprocessing chunks/metadata are incomplete. "
                "Legacy openpose_results.parquet conversion is no longer supported."
            )

        meta_rows = []
        if keep_until > 0 and out_meta.exists():
            old_meta = pd.read_parquet(out_meta)
            if "frame_idx" in old_meta.columns:
                meta_rows = old_meta.loc[old_meta.frame_idx <= keep_until].to_dict("records")

        sb = ShotBoundaryInferencer(ShotBoundaryInferConfig(
            resize_w=self.ctx.sb_cfg["resize"][0], resize_h=self.ctx.sb_cfg["resize"][1],
            grid_x=self.ctx.sb_cfg["grid"][0], grid_y=self.ctx.sb_cfg["grid"][1],
            ema_alpha=self.ctx.sb_cfg.get("ema_alpha", 0.9),
        ))
        if self.ctx.restore_mode and start > 1:
            self.progress.message(f"Preprocessing restore: replaying shot-boundary state to frame {start - 1}/{total}")
            for replay_idx in range(1, start):
                frame_path = frames_dir / f"frame_{replay_idx:06d}.jpg"
                img = cv2.imread(str(frame_path)) if frame_path.exists() else None
                if img is None:
                    _, img = preprocess_image(self.ctx.opWrapper, self.ctx.images[replay_idx - 1], self.ctx.save_width, return_img=True)
                sb.update(img)

        self.start_progress(total)
        if start > 1:
            self.progress_update(completed=start - 1, frame=start - 1, total=total, op="restored", force=True)
            self.progress.message(f"Preprocessing restore: continuing from frame {start}/{total}")

        crops_dir = self.stage_dir / "crops"
        shard_id = self._next_crop_shard(crops_dir)
        since_flush_start = start

        for batch_no, batch in iter_batches(list(range(start - 1, total)), batch_size):
            batch_det_rows, batch_crop_buf = [], []
            batch_start_frame, batch_end_frame = int(batch[0]) + 1, int(batch[-1]) + 1

            for idx in batch:
                datum, img = preprocess_image(self.ctx.opWrapper, self.ctx.images[idx], self.ctx.save_width, return_img=True)
                frame_idx = idx + 1
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), img)

                g = float(sb.update(img))
                reset_mode = bool(g < float(self.ctx.tracker.cfg.reset_g_threshold))
                kps = datum.poseKeypoints
                has_detections = bool(kps is not None and len(kps) > 0)
                meta_rows.append({"frame_idx": frame_idx, "frame_path": str(frame_path), "g": g, "reset_mode": reset_mode, "has_detections": has_detections})

                if has_detections:
                    self._append_openpose_rows(frame_idx, str(frame_path), kps, img, batch_det_rows, batch_crop_buf, shard_id)

                self.progress_update(advance=1, frame=frame_idx, total=total, batch=batch_no, op="OpenPose")

            det_chunk = chunk_path(det_chunks_dir, "openpose", batch_start_frame, batch_end_frame)
            atomic_write_parquet(det_chunk, pd.DataFrame(batch_det_rows, columns=OPENPOSE_RESULTS_COLUMNS))
            if self.ctx.save_log and batch_crop_buf:
                crops_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(crops_dir / f"shard_{shard_id:06d}.npz"), crops=np.asarray(batch_crop_buf, dtype=object))
                shard_id += 1

            should_flush = batch_end_frame - since_flush_start + 1 >= flush_every or batch_end_frame == total
            if should_flush:
                atomic_write_parquet(out_meta, pd.DataFrame(meta_rows, columns=FRAMES_METADATA_COLUMNS))
                self.write_manifest("in_progress", last_flushed_frame=batch_end_frame, last_completed_frame=batch_end_frame, last_chunk_path=str(det_chunk), total_frames=total)
                since_flush_start = batch_end_frame + 1
            free(batch_det_rows, batch_crop_buf)

        atomic_write_parquet(out_meta, pd.DataFrame(meta_rows, columns=FRAMES_METADATA_COLUMNS))
        self.write_manifest("completed", last_flushed_frame=total, last_completed_frame=total, total_frames=total)
        free(meta_rows)

    def _next_crop_shard(self, crops_dir: Path) -> int:
        if not (self.ctx.restore_mode and self.ctx.save_log and crops_dir.exists()):
            return 1
        ids = [int(f.stem.split("_")[-1]) for f in crops_dir.glob("shard_*.npz") if f.stem.split("_")[-1].isdigit()]
        return max(ids) + 1 if ids else 1

    def _append_openpose_rows(self, frame_idx, frame_path, kps, img, rows, crop_buf, shard_id):
        for det_id, person in enumerate(kps):
            conf = person[:, 2].astype(np.float32)
            bbox = keypoints_to_intersection_bbox(person, conf_th=float(self.ctx.tracker.cfg.min_kp_conf), img_w=img.shape[1], img_h=img.shape[0])
            row = {
                "frame_idx": frame_idx, "frame_path": frame_path, "det_id": det_id,
                "keypoints": person[:, :2].astype(np.float32).tolist(), "kp_conf": conf.tolist(),
                "confidence": float(np.nanmean(conf)),
                "bbox_x1": None if bbox is None else float(bbox[0]), "bbox_y1": None if bbox is None else float(bbox[1]),
                "bbox_x2": None if bbox is None else float(bbox[2]), "bbox_y2": None if bbox is None else float(bbox[3]),
                "crop_shard_id": None, "crop_index": None,
            }
            if self.ctx.save_log and bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    crop_buf.append(crop)
                    row.update(crop_shard_id=shard_id, crop_index=len(crop_buf) - 1)
            rows.append(row)


class DetectionPreparationStage(BaseStage):
    name, progress_key, progress_label, progress_step = "detection_preparation", "dp", "Detection Preparation", "[2/6]"

    def run(self) -> None:
        self.prepare_output()
        chunks_dir = self.stage_dir / "prepared_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        meta_path = self.ctx.save_dir / "preprocessed" / "frames_metadata.parquet"
        raw_chunks_dir = self.ctx.save_dir / "preprocessed" / "openpose_chunks"
        self.require(meta_path, "preprocessing metadata")
        if not any(raw_chunks_dir.glob("openpose_*.parquet")):
            raise RuntimeError(f"Missing preprocessing chunks directory: {raw_chunks_dir}")

        meta = pd.read_parquet(meta_path)
        if "frame_idx" not in meta.columns:
            raise RuntimeError("frames_metadata.parquet is missing required column: frame_idx")
        meta_by_frame = first_by_frame(meta)
        frames = sorted(meta_by_frame)
        total, max_frame = len(frames), max(frames) if frames else 0

        manifest = self.manifest()
        if self.restore_completed(total or 1, needed_globs=[(chunks_dir, "prepared_detections_*.parquet")]):
            free(meta)
            return

        start = int(manifest.get("last_completed_frame", 0) or 0) + 1 if self.ctx.restore_mode else 1
        if not frames:
            self.start_progress(1)
            self.progress_update(completed=1, frame=0, total=0, op="no frames", force=True)
            self.write_manifest("completed", last_completed_frame=0, total_frames=0)
            free(meta)
            return
        if self.ctx.restore_mode and start > max_frame:
            self.start_progress(total, " | op=RESTORE | dev=IO")
            self.progress_update(completed=total, frame=max_frame, total=max_frame, op="restored complete", force=True)
            self.write_manifest("completed", last_completed_frame=max_frame, total_frames=total)
            free(meta)
            return

        batch_size = self.cfg_int("detection_preparation", "batch_size", 128)
        self.start_progress(total)
        if start > 1:
            self.progress_update(completed=start - 1, frame=start - 1, total=max_frame, op="restored", force=True)

        for batch_no, batch in iter_batches([f for f in frames if f >= start], batch_size):
            raw_batch = read_chunked_parquet_for_frames(
                raw_chunks_dir,
                "openpose",
                batch,
                columns=["frame_idx", "keypoints", "kp_conf"],
                expected_columns=["frame_idx", "keypoints", "kp_conf"],
            )
            raw_by_frame = df_by_frame(raw_batch)
            prepared_rows = []

            for frame_idx in batch:
                meta_row = meta_by_frame.get(frame_idx)
                if meta_row is None:
                    continue
                img = cv2.imread(str(meta_row.frame_path))
                if img is None:
                    raise RuntimeError(f"Failed to read preprocessed frame: {meta_row.frame_path}")

                detections = self._prepare_frame(raw_by_frame.get(frame_idx, pd.DataFrame()), img)
                prepared_rows.extend(detections_to_prepared_rows(detections, frame_idx))
                self.progress_update(advance=1, frame=frame_idx, total=max_frame, batch=batch_no, op="prepare detections")

            batch_start, batch_end = int(batch[0]), int(batch[-1])
            out_chunk = chunk_path(chunks_dir, "prepared_detections", batch_start, batch_end)
            atomic_write_parquet(out_chunk, pd.DataFrame(prepared_rows, columns=PREPARED_DETECTIONS_COLUMNS))
            self.write_manifest("in_progress", last_completed_frame=batch_end, last_chunk_path=str(out_chunk), total_frames=total)
            free(raw_batch, raw_by_frame, prepared_rows)

        self.write_manifest("completed", last_completed_frame=max_frame, total_frames=total)
        free(meta)

    def _prepare_frame(self, frame_rows: pd.DataFrame, img: np.ndarray):
        if frame_rows is None or frame_rows.empty:
            return []
        persons = []
        for row in frame_rows.itertuples(index=False):
            xy, cf = normalize_keypoints_xy(row.keypoints), normalize_kp_conf(row.kp_conf)
            k = min(xy.shape[0], cf.shape[0])
            if k > 0:
                persons.append(np.concatenate([xy[:k], cf[:k, None]], axis=1).astype(np.float32, copy=False))
        if not persons:
            return []
        return prepare_frame_detections_from_keypoints(
            kps=np.stack(persons).astype(np.float32, copy=False),
            original_img=img,
            conf_th=self.ctx.tracker.cfg.min_kp_conf,
            tracker=self.ctx.tracker,
            app_embedder=self.ctx.app_embedder,
            select_top_with_nearest=self.ctx.select_top_with_nearest,
            extract_features_with_hsv=self.ctx.extract_features_with_hsv,
            build_fused_appearance_embedding_with_mask=self.ctx.build_fused_appearance_embedding_with_mask,
        )


class LocalTrackingStage(BaseStage):
    name, progress_key, progress_label, progress_step = "local_tracking", "lt", "Local Tracking", "[3/6]"

    def run(self) -> None:
        self.prepare_output()
        ck, chunks_dir, states_dir, logs_dir = [self.stage_dir / p for p in ("checkpoints", "chunks", "track_states", "logs")]
        for d in (ck, chunks_dir, states_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.ctx.save_log:
            logs_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = ck / "state_latest.pkl"
        manifest = self.manifest()
        if self.restore_completed(int(manifest.get("total_frames", 1) or 1), needed_paths=[checkpoint_path], needed_globs=[(chunks_dir, "local_tracks_*.parquet")]):
            return

        start_frame = 1
        if self.ctx.restore_mode and checkpoint_path.exists():
            self.ctx.tracker, checkpoint_frame = load_tracker_checkpoint(checkpoint_path)
            start_frame = checkpoint_frame + 1
        elif self.ctx.restore_mode and int(manifest.get("last_checkpoint_frame", manifest.get("last_completed_frame", 0)) or 0) > 0:
            raise RuntimeError("Restore requested, but local_tracking/checkpoints/state_latest.pkl is missing.")

        meta_path = self.ctx.save_dir / "preprocessed" / "frames_metadata.parquet"
        prepared_chunks_dir = self.ctx.save_dir / "detection_preparation" / "prepared_chunks"
        self.require(meta_path, "preprocessing metadata")
        if not any(prepared_chunks_dir.glob("prepared_detections_*.parquet")):
            raise RuntimeError(f"Missing detection preparation chunks directory: {prepared_chunks_dir}")

        meta = pd.read_parquet(meta_path)
        if "frame_idx" not in meta.columns:
            raise RuntimeError("frames_metadata.parquet is missing required column: frame_idx")
        meta_by_frame = first_by_frame(meta)
        frame_ids = sorted(meta_by_frame)
        total, max_frame = len(frame_ids), max(frame_ids) if frame_ids else 0
        if not frame_ids:
            self.start_progress(1)
            self.progress_update(completed=1, frame=0, total=0, op="no frames", force=True)
            self.write_manifest("completed", last_flushed_frame=0, last_checkpoint_frame=0, last_completed_frame=0, total_frames=0)
            return
        if self.ctx.restore_mode and start_frame > max_frame:
            self.start_progress(total, " | op=RESTORE | dev=IO")
            self.progress_update(completed=total, frame=max_frame, total=max_frame, op="restored complete", force=True)
            self.write_manifest("completed", last_flushed_frame=max_frame, last_checkpoint_frame=max_frame, last_completed_frame=max_frame, total_frames=total)
            return

        batch_size, ck_every = self.cfg_int("local_tracking", "batch_size", 128), self.cfg_int("local_tracking", "checkpoint_every_frames", 500)
        self.start_progress(total)
        if start_frame > 1:
            self.progress_update(completed=start_frame - 1, frame=start_frame - 1, total=max_frame, op="restored", force=True)

        for batch_no, batch in iter_batches([f for f in frame_ids if f >= start_frame], batch_size):
            prepared_batch = read_chunked_parquet_for_frames(
                prepared_chunks_dir,
                "prepared_detections",
                batch,
                columns=PREPARED_DETECTIONS_COLUMNS,
                expected_columns=PREPARED_DETECTIONS_COLUMNS,
            )
            prepared_by_frame = df_by_frame(prepared_batch)
            track_rows, state_rows, log_rows = [], [], []

            for frame_idx in batch:
                meta_row = meta_by_frame.get(frame_idx)
                if meta_row is None:
                    continue
                img = cv2.imread(str(meta_row.frame_path))
                if img is None:
                    raise RuntimeError(f"Failed to read preprocessed frame: {meta_row.frame_path}")
                detections = prepared_rows_to_detections(prepared_by_frame.get(frame_idx, pd.DataFrame()))
                detections, log = update_tracker_from_detections(
                    detections=detections,
                    tracker=self.ctx.tracker,
                    g=float(meta_row.g),
                    reset_mode=bool(meta_row.reset_mode),
                )
                matches = [(int(tid), int(did)) for tid, did in (log or {}).get("matches", [])]
                epoch_id = int((log or {}).get("epoch_id", 1))
                track_rows.extend({"frame_idx": int(frame_idx), "epoch_id": epoch_id, "local_track_id": int(tid), "det_id": int(did)} for tid, did in matches)
                state_rows.extend(collect_tracker_state_rows(tracker=self.ctx.tracker, frame_idx=frame_idx, epoch_id=epoch_id, matches=matches, img_shape=img.shape, detections=detections or []))
                if self.ctx.save_log:
                    log_rows.append({"frame_idx": frame_idx, "epoch_id": epoch_id, "g": float(meta_row.g), "reset_mode": bool(meta_row.reset_mode), "log_json": json.dumps(log or {}, default=str)})
                self.progress_update(advance=1, frame=frame_idx, total=max_frame, batch=batch_no)

            batch_start, batch_end = int(batch[0]), int(batch[-1])
            local_chunk = chunk_path(chunks_dir, "local_tracks", batch_start, batch_end)
            state_chunk = chunk_path(states_dir, "track_states", batch_start, batch_end)
            atomic_write_parquet(local_chunk, pd.DataFrame(track_rows, columns=LOCAL_TRACKS_COLUMNS))
            atomic_write_parquet(state_chunk, pd.DataFrame(state_rows, columns=TRACK_STATES_COLUMNS))
            if self.ctx.save_log:
                atomic_write_parquet(chunk_path(logs_dir, "tracking_logs", batch_start, batch_end), pd.DataFrame(log_rows, columns=TRACKING_LOGS_COLUMNS))
            if hasattr(self.ctx.tracker, "compact_memory"):
                self.ctx.tracker.compact_memory()
            save_tracker_checkpoint(checkpoint_path, self.ctx.tracker, batch_end)
            if ck_every > 0 and (batch_end % ck_every == 0 or batch_end == max_frame):
                save_tracker_checkpoint(ck / f"state_frame_{batch_end:06d}.pkl", self.ctx.tracker, batch_end)
            self.write_manifest("in_progress", last_flushed_frame=batch_end, last_checkpoint_frame=batch_end, last_completed_frame=batch_end, last_chunk_path=str(local_chunk), last_state_chunk_path=str(state_chunk), total_frames=total)
            free(prepared_batch, prepared_by_frame, track_rows, state_rows, log_rows)

        save_tracker_checkpoint(checkpoint_path, self.ctx.tracker, max_frame)
        self.write_manifest("completed", last_flushed_frame=max_frame, last_checkpoint_frame=max_frame, last_completed_frame=max_frame, total_frames=total)
        free(meta)


class LocalDetSavingStage(BaseStage):
    name, progress_key, progress_label, progress_step = "local_track_saving", "lds", "Local Det Saving", "[4/6]"

    def run(self) -> None:
        self.prepare_output()
        matched_dir, unmatched_dir, frames_dir = [self.stage_dir / p for p in ("matched_dets", "unmatched_dets", "frames")]
        for d in (matched_dir, unmatched_dir, frames_dir):
            d.mkdir(parents=True, exist_ok=True)
        det_chunks_dir, meta_path, local_chunks_dir = self._required_inputs()
        meta = pd.read_parquet(meta_path)
        meta_by_frame = first_by_frame(meta)
        frames = sorted(meta_by_frame)
        max_frame = max(frames) if frames else 0

        start = self._restore_start(frames_dir, frames, max_frame)
        if start is None:
            return

        self.start_progress(len(frames), " | op=START | dev=IO")
        if start > 1:
            self.progress_update(completed=start - 1, frame=start - 1, total=max_frame, op="restored", force=True)
        bs = self.cfg_int("local_det_saving", "batch_size", 128)

        for batch_no, batch in iter_batches([f for f in frames if f >= start], bs):
            self.progress_update(completed=max(batch[0] - 1, 0), frame=batch[0], total=max_frame, batch=batch_no, op="read chunks", force=True)
            det_batch = read_chunked_parquet_for_frames(det_chunks_dir, "prepared_detections", batch, columns=PREPARED_DETECTIONS_COLUMNS, expected_columns=PREPARED_DETECTIONS_COLUMNS)
            local_batch = read_chunked_parquet_for_frames(local_chunks_dir, "local_tracks", batch, columns=LOCAL_TRACKS_COLUMNS, expected_columns=LOCAL_TRACKS_COLUMNS)
            det_by_frame, local_by_frame = df_by_frame(det_batch), df_by_frame(local_batch)
            match_idx = {(int(r.frame_idx), int(r.det_id)): int(r.local_track_id) for r in local_batch.itertuples(index=False)}

            for frame_idx in batch:
                mrow = meta_by_frame.get(frame_idx)
                if mrow is None:
                    continue
                img = cv2.imread(str(mrow.frame_path))
                if img is None:
                    raise RuntimeError(f"Failed to read preprocessed frame: {mrow.frame_path}")
                detections, row_by_det_id, vis_idx_by_det_id = build_visual_detections_from_rows(det_by_frame.get(frame_idx, pd.DataFrame()), img)
                self._write_crops(frame_idx, img, detections, row_by_det_id, vis_idx_by_det_id, match_idx, matched_dir, unmatched_dir)
                local_matches = self._local_matches(local_by_frame.get(frame_idx, pd.DataFrame()), vis_idx_by_det_id)
                local_frame = img.copy()
                render_tracking_overlays(frame=local_frame, detections=detections, matches=local_matches, frame_idx=frame_idx, use_global_ids=False)
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), local_frame)
                self.write_manifest("in_progress", last_saved_frame=frame_idx, total_frames=len(frames))
                self.progress_update(advance=1, frame=frame_idx, total=max_frame, batch=batch_no, op="write local frame")
            free(det_batch, local_batch, det_by_frame, local_by_frame, match_idx)

        self.write_manifest("completed", last_saved_frame=max(frames) if frames else 0, total_frames=len(frames))
        free(meta)

    def _required_inputs(self):
        det_chunks_dir = self.ctx.save_dir / "detection_preparation" / "prepared_chunks"
        meta_path = self.ctx.save_dir / "preprocessed" / "frames_metadata.parquet"
        local_chunks_dir = self.ctx.save_dir / "local_tracking" / "chunks"
        if not any(det_chunks_dir.glob("prepared_detections_*.parquet")):
            raise RuntimeError(f"Missing detection preparation chunks directory: {det_chunks_dir}")
        self.require(local_chunks_dir, "local tracking chunks directory")
        self.require(meta_path, "preprocessing metadata")
        return det_chunks_dir, meta_path, local_chunks_dir

    def _restore_start(self, frames_dir: Path, frames: list[int], max_frame: int) -> int | None:
        manifest = self.manifest()
        if self.ctx.restore_mode and manifest.get("status") == "completed":
            total = int(manifest.get("total_frames", len(frames)) or len(frames) or 1)
            if len(list(frames_dir.glob("frame_*.jpg"))) >= max(1, min(total, len(frames) or total)):
                self.start_progress(total, " | op=RESTORE | dev=IO")
                self.progress_update(completed=total, frame=total, total=total, op="restored complete", force=True)
                return None
            self.progress.warning("Local det saving manifest completed, but rendered frames are missing; rebuilding from frame 1")
            return 1
        start = int(manifest.get("last_saved_frame", 0)) + 1 if self.ctx.restore_mode else 1
        if self.ctx.restore_mode and frames and start > max_frame:
            if len(list(frames_dir.glob("frame_*.jpg"))) < len(frames):
                self.progress.warning("Local rendered frames are incomplete; rebuilding from frame 1")
                return 1
            self.start_progress(len(frames), " | op=RESTORE | dev=IO")
            self.progress_update(completed=len(frames), frame=max_frame, total=max_frame, op="restored complete", force=True)
            return None
        return start

    @staticmethod
    def _local_matches(local_rows: pd.DataFrame, vis_idx_by_det_id: dict[int, int]):
        if local_rows is None or local_rows.empty:
            return []
        matches = []
        for r in local_rows.itertuples(index=False):
            vis_idx = vis_idx_by_det_id.get(int(r.det_id))
            if vis_idx is not None:
                matches.append((int(r.local_track_id), int(vis_idx)))
        return matches

    @staticmethod
    def _write_crops(frame_idx, img, detections, row_by_det_id, vis_idx_by_det_id, match_idx, matched_dir, unmatched_dir):
        for det_id in row_by_det_id:
            det_vis_idx = vis_idx_by_det_id.get(det_id)
            if det_vis_idx is None or det_vis_idx >= len(detections):
                continue
            bbox = detections[det_vis_idx].meta.get("raw", {}).get("bbox")
            if bbox is None:
                continue
            tid = match_idx.get((frame_idx, int(det_id)))
            x1, y1, x2, y2 = map(int, bbox)
            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size:
                name = f"frame_{frame_idx:06d}_det_{int(det_id):03d}.jpg" if tid is None else f"frame_{frame_idx:06d}_track_{tid:03d}_det_{int(det_id):03d}.jpg"
                cv2.imwrite(str((unmatched_dir if tid is None else matched_dir) / name), crop)


class GlobalClusteringStage(BaseStage):
    name, progress_key, progress_label, progress_step = "global_clustering", "gc", "Global Clustering", "[5/6]"

    def run(self) -> None:
        self.prepare_output()
        checkpoints_dir = self.stage_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        if self.restore_completed(int(manifest.get("total_tracks", 1) or 1), needed_paths=[self.stage_dir / "local_to_global.parquet"]):
            return

        local_ck = self.ctx.save_dir / "local_tracking" / "checkpoints" / "state_latest.pkl"
        self.require(local_ck, "local_tracking/checkpoints/state_latest.pkl for global clustering")
        self.start_progress(1)
        loaded_tracker, _ = load_tracker_checkpoint(local_ck)
        if not hasattr(loaded_tracker, "get_epoch_tracks"):
            raise RuntimeError("Loaded tracker checkpoint has no get_epoch_tracks()")
        epoch_tracks = loaded_tracker.get_epoch_tracks()
        if sum(len(v) for v in epoch_tracks.values()) == 0:
            self._write_empty_outputs()
            return

        params = self.ctx.graph_clustering_params or {}
        clusterer = GlobalTrackClusterer(
            k=int(params.get("k", 5)), sim_threshold=float(params.get("sim_threshold", 0.5)),
            n_clusters=int(params.get("n_clusters", 2)), random_state=int(params.get("random_state", 42)),
            assign_labels=str(params.get("assign_labels", "kmeans")),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Graph is not fully connected.*", category=UserWarning)
            mapping, sim = clusterer.build_mapping(epoch_tracks=epoch_tracks, return_similarity=True)
        rows = [{"epoch_id": int(e), "local_track_id": int(l), "global_track_id": int(g)} for (e, l), g in mapping.items()]
        df = pd.DataFrame(rows, columns=GLOBAL_MAP_COLUMNS)
        atomic_write_parquet(self.stage_dir / "local_to_global.parquet", df)
        atomic_write_parquet(self.stage_dir / "global_clusters.parquet", df)
        np.savez_compressed(str(self.stage_dir / "tracks_similarity.npz"), sim=sim["sim"], nodes=np.asarray(sim["nodes"], dtype=np.int32))
        save_tracker_checkpoint(checkpoints_dir / "state_latest.pkl", loaded_tracker, len(rows) - 1)
        self.write_manifest("completed", last_completed_track=len(rows) - 1, total_tracks=len(rows))
        self.progress_update(completed=1, frame=1, total=1, batch=1, force=True)
        free(loaded_tracker, epoch_tracks, mapping, sim, rows)

    def _write_empty_outputs(self):
        empty = pd.DataFrame(columns=GLOBAL_MAP_COLUMNS)
        atomic_write_parquet(self.stage_dir / "local_to_global.parquet", empty)
        atomic_write_parquet(self.stage_dir / "global_clusters.parquet", empty)
        np.savez_compressed(str(self.stage_dir / "tracks_similarity.npz"), sim=np.zeros((0, 0), dtype=np.float32), nodes=np.zeros((0, 2), dtype=np.int32))
        self.write_manifest("completed", last_completed_track=-1, total_tracks=0)
        self.progress_update(completed=1, frame=1, total=1, batch=1, force=True)


class GlobalSavingStage(BaseStage):
    name, progress_key, progress_label, progress_step = "global_saving", "gs", "Global Saving", "[6/6]"

    def run(self) -> None:
        self.prepare_output()
        frames_dir, debug_dir = self.stage_dir / "frames", self.stage_dir / "debug_logs"
        frames_dir.mkdir(parents=True, exist_ok=True)
        if self.ctx.save_log:
            debug_dir.mkdir(parents=True, exist_ok=True)

        det_chunks_dir, meta_path, local_chunks_dir, mapping_path = self._required_inputs()
        meta = pd.read_parquet(meta_path)
        meta_by_frame = first_by_frame(meta)
        frames = sorted(meta_by_frame)
        max_frame = max(frames) if frames else 0
        mapping = pd.read_parquet(mapping_path)
        map_idx = {(int(r.epoch_id), int(r.local_track_id)): int(r.global_track_id) for r in mapping.itertuples(index=False)}

        manifest = self.manifest()
        start = int(manifest.get("last_saved_frame", 0)) + 1 if self.ctx.restore_mode else 1
        self.start_progress(len(frames), " | op=START | dev=IO")
        if self.ctx.restore_mode and frames and start > max_frame:
            self.progress_update(completed=len(frames), frame=max_frame, total=max_frame, op="restored complete", force=True)
            self.write_manifest("completed", last_saved_frame=max_frame, total_frames=len(frames))
            free(meta, mapping, map_idx)
            return
        if start > 1:
            self.progress_update(completed=start - 1, frame=start - 1, total=max_frame, op="restored", force=True)

        for batch_no, batch in iter_batches([f for f in frames if f >= start], self.cfg_int("global_saving", "batch_size", 128)):
            det_batch = read_chunked_parquet_for_frames(det_chunks_dir, "prepared_detections", batch, columns=PREPARED_DETECTIONS_COLUMNS, expected_columns=PREPARED_DETECTIONS_COLUMNS)
            local_batch = read_chunked_parquet_for_frames(local_chunks_dir, "local_tracks", batch, columns=LOCAL_TRACKS_COLUMNS, expected_columns=LOCAL_TRACKS_COLUMNS)
            det_by_frame, local_by_frame = df_by_frame(det_batch), df_by_frame(local_batch)

            for frame_idx in batch:
                mrow = meta_by_frame.get(frame_idx)
                if mrow is None:
                    continue
                img = cv2.imread(str(mrow.frame_path))
                if img is None:
                    raise RuntimeError(f"Failed to read preprocessed frame: {mrow.frame_path}")
                detections, _, vis_idx_by_det_id = build_visual_detections_from_rows(det_by_frame.get(frame_idx, pd.DataFrame()), img)
                matches = self._global_matches(local_by_frame.get(frame_idx, pd.DataFrame()), vis_idx_by_det_id, map_idx)
                render_tracking_overlays(frame=img, detections=detections, matches=matches, frame_idx=frame_idx, use_global_ids=True)
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), img)
                if self.ctx.save_log:
                    (debug_dir / f"frame_{frame_idx:06d}.json").write_text(json.dumps({"frame_idx": frame_idx, "global_matches": matches}), encoding="utf-8")
                self.write_manifest("in_progress", last_saved_frame=frame_idx, total_frames=len(frames))
                self.progress_update(advance=1, frame=frame_idx, total=max_frame, batch=batch_no)
            free(det_batch, local_batch, det_by_frame, local_by_frame)

        self.write_manifest("completed", last_saved_frame=max(frames) if frames else 0, total_frames=len(frames))
        free(meta, mapping, map_idx)

    def _required_inputs(self):
        det_chunks_dir = self.ctx.save_dir / "detection_preparation" / "prepared_chunks"
        meta_path = self.ctx.save_dir / "preprocessed" / "frames_metadata.parquet"
        local_chunks_dir = self.ctx.save_dir / "local_tracking" / "chunks"
        mapping_path = self.ctx.save_dir / "global_clustering" / "local_to_global.parquet"
        if not any(det_chunks_dir.glob("prepared_detections_*.parquet")):
            raise RuntimeError(f"Missing detection preparation chunks directory: {det_chunks_dir}")
        self.require(local_chunks_dir, "local tracking chunks directory")
        self.require(meta_path, "preprocessing metadata")
        self.require(mapping_path, "global mapping")
        return det_chunks_dir, meta_path, local_chunks_dir, mapping_path

    @staticmethod
    def _global_matches(local_rows: pd.DataFrame, vis_idx_by_det_id: dict[int, int], map_idx: dict[tuple[int, int], int]):
        if local_rows is None or local_rows.empty:
            return []
        matches = []
        for r in local_rows.itertuples(index=False):
            gid = map_idx.get((int(r.epoch_id), int(r.local_track_id)))
            vis_idx = vis_idx_by_det_id.get(int(r.det_id))
            if gid is not None and vis_idx is not None:
                matches.append((int(gid), int(vis_idx)))
        return matches