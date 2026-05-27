"""Staged tracking pipeline implementation with resumable manifests/checkpoints."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from boxing_project.tracking.frame_processing import prepare_frame_detections_from_keypoints, update_tracker_from_detections
from boxing_project.tracking.global_clustering import GlobalTrackClusterer
from boxing_project.tracking.image_utils import keypoints_to_intersection_bbox, render_tracking_overlays
from boxing_project.tracking.openpose_processing import preprocess_image




OPENPOSE_RESULTS_COLUMNS = [
    "frame_idx", "frame_path", "det_id", "keypoints", "kp_conf", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "crop_shard_id", "crop_index",
]
FRAMES_METADATA_COLUMNS = ["frame_idx", "frame_path", "g", "reset_mode", "has_detections"]
LOCAL_TRACKS_COLUMNS = ["frame_idx", "epoch_id", "local_track_id", "det_id"]
TRACKING_LOGS_COLUMNS = ["frame_idx", "epoch_id", "g", "reset_mode", "log_json"]
GLOBAL_MAP_COLUMNS = ["epoch_id", "local_track_id", "global_track_id"]

class _LightDet:
    """Small detection-like object compatible with render_tracking_overlays."""

    def __init__(self, bbox, keypoints, kp_conf):
        self.meta = {"raw": {"bbox": bbox}}
        self.center = ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)
        self.keypoints = keypoints
        self.kp_conf = kp_conf


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_batches(items: list[int], batch_size: int):
    size = max(1, int(batch_size))
    for i in range(0, len(items), size):
        yield i // size + 1, items[i:i + size]


def save_tracker_checkpoint(path: Path, tracker: Any, last_completed_frame: int) -> None:
    payload = {"mode": "pickle_tracker", "tracker": tracker, "last_completed_frame": int(last_completed_frame)}
    try:
        atomic_write_bytes(path, pickle.dumps(payload))
    except Exception as exc:
        raise RuntimeError("Failed to serialize tracker checkpoint; implement explicit tracker state serialization.") from exc


def load_tracker_checkpoint(path: Path) -> tuple[Any, int]:
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception as exc:
        raise RuntimeError(f"Corrupt tracker checkpoint: {path}") from exc
    if payload.get("mode") != "pickle_tracker":
        raise RuntimeError(f"Unsupported checkpoint mode: {payload.get('mode')}")
    return payload["tracker"], int(payload.get("last_completed_frame", -1))


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


class RichStageProgress:
    def __init__(self, enabled: bool = True):
        self._progress = None
        self._tasks = {}
        if enabled:
            self._progress = Progress(TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn())
            self._progress.start()

    def add(self, key: str, desc: str, total: int) -> None:
        if self._progress is not None and key not in self._tasks:
            self._tasks[key] = self._progress.add_task(desc, total=total)

    def update(self, key: str, **kwargs) -> None:
        if self._progress is not None and key in self._tasks:
            self._progress.update(self._tasks[key], **kwargs)

    def finish(self) -> None:
        if self._progress is not None:
            self._progress.stop()


class BaseStage:
    name = "base"

    def __init__(self, ctx: PipelineContext, progress: RichStageProgress):
        self.ctx = ctx
        self.progress = progress
        self.stage_dir = self.ctx.save_dir / self.name
        self.manifest_path = self.stage_dir / "manifest.json"

    def prepare_output(self) -> None:
        if not self.ctx.restore_mode and self.stage_dir.exists():
            shutil.rmtree(self.stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)


class PreprocessingStage(BaseStage):
    name = "preprocessed"

    def run(self) -> None:
        self.prepare_output()
        from boxing_project.shot_boundary.inference import ShotBoundaryInferConfig, ShotBoundaryInferencer

        frames_dir = self.stage_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        manifest = load_manifest(self.manifest_path)
        out_det = self.stage_dir / "openpose_results.parquet"
        out_meta = self.stage_dir / "frames_metadata.parquet"
        if self.ctx.restore_mode and manifest.get("status") == "completed":
            if out_meta.exists() and out_det.exists():
                try:
                    cols = set(pd.read_parquet(out_meta).columns.tolist())
                    if set(FRAMES_METADATA_COLUMNS).issubset(cols):
                        return
                except Exception:
                    pass

        cfg = self.ctx.cfg.get("preprocessing", {})
        batch_size = int(cfg.get("batch_size", 32))
        shard_size = int(cfg.get("checkpoint_every_frames", 100))


        start = 1
        keep_until = 0
        if self.ctx.restore_mode:
            if out_meta.exists():
                prev_meta = pd.read_parquet(out_meta)
                if not prev_meta.empty and "frame_idx" in prev_meta.columns:
                    keep_until = int(prev_meta["frame_idx"].max())
                start = keep_until + 1
            else:
                keep_until = 0
                start = 1

        det_rows = []
        meta_rows = []
        if self.ctx.restore_mode and keep_until >= 0:
            if out_det.exists():
                old = pd.read_parquet(out_det)
                det_rows = old.loc[old.frame_idx <= keep_until].to_dict("records")
            if out_meta.exists():
                oldm = pd.read_parquet(out_meta)
                meta_rows = oldm.loc[oldm.frame_idx <= keep_until].to_dict("records")

        sb = ShotBoundaryInferencer(ShotBoundaryInferConfig(
            resize_w=self.ctx.sb_cfg["resize"][0], resize_h=self.ctx.sb_cfg["resize"][1],
            grid_x=self.ctx.sb_cfg["grid"][0], grid_y=self.ctx.sb_cfg["grid"][1],
            ema_alpha=self.ctx.sb_cfg.get("ema_alpha", 0.9),
        ))
        # restore shot-boundary state by replaying previous frames
        if self.ctx.restore_mode and start > 0:
            replay_to = start - 1
            for ridx in range(1, replay_to + 1):
                frame_path = frames_dir / f"frame_{ridx:06d}.jpg"
                if frame_path.exists():
                    img = cv2.imread(str(frame_path))
                else:
                    _, img = preprocess_image(self.ctx.opWrapper, self.ctx.images[ridx-1], self.ctx.save_width, return_img=True)
                sb.update(img)

        self.progress.add("pre", "[1/5] Preprocessing", total=max(len(self.ctx.images), 1))
        if start > 0:
            self.progress.update("pre", completed=max(0, start-1))

        crops_dir = self.stage_dir / "crops"
        crop_buf: list[np.ndarray] = []
        shard_id = 1
        if self.ctx.restore_mode and self.ctx.save_log and crops_dir.exists():
            existing_ids = []
            for f in crops_dir.glob("shard_*.npz"):
                try:
                    existing_ids.append(int(f.stem.split("_")[-1]))
                except Exception:
                    pass
            if existing_ids:
                shard_id = max(existing_ids) + 1

        for batch_no, batch in iter_batches(list(range(start-1, len(self.ctx.images))), batch_size):
            for idx in batch:
                datum, img = preprocess_image(self.ctx.opWrapper, self.ctx.images[idx], self.ctx.save_width, return_img=True)
                frame_idx = idx + 1
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), img)

                g = float(sb.update(img))
                reset_mode = bool(g < float(self.ctx.tracker.cfg.reset_g_threshold))

                kps = datum.poseKeypoints
                has_detections = bool(kps is not None and len(kps) > 0)
                meta_rows.append({
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path),
                    "g": g,
                    "reset_mode": reset_mode,
                    "has_detections": has_detections,
                })

                if has_detections:
                    for det_id, person in enumerate(kps):
                        conf = person[:, 2].astype(np.float32)
                        bbox = keypoints_to_intersection_bbox(person, conf_th=float(self.ctx.tracker.cfg.min_kp_conf), img_w=img.shape[1], img_h=img.shape[0])
                        row = {
                            "frame_idx": frame_idx,
                            "frame_path": str(frame_path),
                            "det_id": det_id,
                            "keypoints": person[:, :2].astype(np.float32).tolist(),
                            "kp_conf": conf.tolist(),
                            "confidence": float(np.nanmean(conf)),
                            "bbox_x1": None if bbox is None else float(bbox[0]),
                            "bbox_y1": None if bbox is None else float(bbox[1]),
                            "bbox_x2": None if bbox is None else float(bbox[2]),
                            "bbox_y2": None if bbox is None else float(bbox[3]),
                        }
                        if self.ctx.save_log and bbox is not None:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                            if crop.size:
                                crop_buf.append(crop)
                                row["crop_shard_id"] = shard_id
                                row["crop_index"] = len(crop_buf) - 1
                        det_rows.append(row)

                if self.ctx.save_log and len(crop_buf) >= shard_size:
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(str(crops_dir / f"shard_{shard_id:06d}.npz"), crops=np.asarray(crop_buf, dtype=object))
                    crop_buf = []
                    shard_id += 1

                if frame_idx % shard_size == 0:
                    atomic_write_parquet(out_det, pd.DataFrame(det_rows, columns=OPENPOSE_RESULTS_COLUMNS))
                    atomic_write_parquet(out_meta, pd.DataFrame(meta_rows, columns=FRAMES_METADATA_COLUMNS))
                    atomic_write_json(self.manifest_path, {"stage": self.name, "status": "in_progress", "last_flushed_frame": frame_idx, "last_completed_frame": frame_idx, "total_frames": len(self.ctx.images)})
                self.progress.update("pre", advance=1, description=f"[1/5] Preprocessing batch {batch_no}")

        if self.ctx.save_log and crop_buf:
            crops_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(crops_dir / f"shard_{shard_id:06d}.npz"), crops=np.asarray(crop_buf, dtype=object))

        atomic_write_parquet(out_det, pd.DataFrame(det_rows, columns=OPENPOSE_RESULTS_COLUMNS))
        atomic_write_parquet(out_meta, pd.DataFrame(meta_rows, columns=FRAMES_METADATA_COLUMNS))
        atomic_write_json(self.manifest_path, {"stage": self.name, "status": "completed", "last_flushed_frame": len(self.ctx.images), "last_completed_frame": len(self.ctx.images), "total_frames": len(self.ctx.images)})


class LocalTrackingStage(BaseStage):
    name = "local_tracking"

    def run(self) -> None:
        self.prepare_output()
        ck = self.stage_dir / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)

        manifest = load_manifest(self.manifest_path)
        checkpoint_path = ck / "state_latest.pkl"
        tracks_out = self.stage_dir / "local_tracks.parquet"
        logs_out = self.stage_dir / "tracking_logs.parquet"

        if self.ctx.restore_mode and manifest.get("status") == "completed":
            if checkpoint_path.exists() and tracks_out.exists():
                return

        checkpoint_loaded = False
        checkpoint_frame = 0
        if self.ctx.restore_mode and checkpoint_path.exists():
            # checkpoint is the source of truth for resume position
            self.ctx.tracker, checkpoint_frame = load_tracker_checkpoint(checkpoint_path)
            checkpoint_loaded = True
            if not tracks_out.exists():
                raise RuntimeError("Restore checkpoint exists but local_tracks.parquet is missing; cannot safely resume local tracking.")
            if self.ctx.save_log and not logs_out.exists():
                raise RuntimeError("Restore checkpoint exists but tracking_logs.parquet is missing while save_log=true.")

        if checkpoint_loaded:
            start_frame = checkpoint_frame + 1
        elif self.ctx.restore_mode:
            parquet_max = 0
            if tracks_out.exists():
                old_t = pd.read_parquet(tracks_out)
                if not old_t.empty and "frame_idx" in old_t.columns:
                    parquet_max = int(old_t["frame_idx"].max())
            if self.ctx.save_log and logs_out.exists():
                old_l = pd.read_parquet(logs_out)
                if not old_l.empty and "frame_idx" in old_l.columns:
                    parquet_max = max(parquet_max, int(old_l["frame_idx"].max()))
            start_frame = parquet_max + 1
        else:
            start_frame = 1

        pre = pd.read_parquet(self.ctx.save_dir / "preprocessed" / "openpose_results.parquet")
        meta = pd.read_parquet(self.ctx.save_dir / "preprocessed" / "frames_metadata.parquet")
        frame_ids = sorted(int(v) for v in meta.frame_idx.unique().tolist())

        cfg = self.ctx.cfg.get("local_tracking", {})
        batch_size = int(cfg.get("batch_size", 128))
        ck_every = int(cfg.get("checkpoint_every_frames", 500))

        track_rows = []
        log_rows = []
        if self.ctx.restore_mode and tracks_out.exists() and start_frame > 1:
            old = pd.read_parquet(tracks_out)
            limit = checkpoint_frame if checkpoint_loaded else (start_frame - 1)
            track_rows = old.loc[old.frame_idx <= limit].to_dict("records")
        if self.ctx.save_log and self.ctx.restore_mode and logs_out.exists() and start_frame > 1:
            old_logs = pd.read_parquet(logs_out)
            limit = checkpoint_frame if checkpoint_loaded else (start_frame - 1)
            log_rows = old_logs.loc[old_logs.frame_idx <= limit].to_dict("records")

        self.progress.add("lt", "[2/5] Local Tracking", total=max(len(frame_ids), 1))
        if start_frame > 1:
            self.progress.update("lt", completed=start_frame - 1)

        pending = [f for f in frame_ids if f >= start_frame]
        for batch_no, batch in iter_batches(pending, batch_size):
            for frame_idx in batch:
                meta_row = meta.loc[meta.frame_idx == frame_idx].iloc[0]
                img = cv2.imread(str(meta_row.frame_path))
                frame_rows = pre.loc[pre.frame_idx == frame_idx]

                if len(frame_rows) == 0:
                    # Explicitly advance tracker on zero-detection frames.
                    log = self.ctx.tracker.update([], g=float(meta_row.g), reset_mode=bool(meta_row.reset_mode))
                else:
                    person_kps = []
                    for _, row in frame_rows.iterrows():
                        xy = np.asarray(row.keypoints, dtype=np.float32)
                        cf = np.asarray(row.kp_conf, dtype=np.float32).reshape(-1)
                        if xy.ndim == 2 and xy.shape[1] == 2 and cf.shape[0] == xy.shape[0]:
                            person_kps.append(np.concatenate([xy, cf[:, None]], axis=1))
                    kps = np.stack(person_kps).astype(np.float32) if person_kps else np.empty((0, 25, 3), dtype=np.float32)
                    dets = prepare_frame_detections_from_keypoints(
                        kps=kps,
                        original_img=img,
                        conf_th=self.ctx.tracker.cfg.min_kp_conf,
                        tracker=self.ctx.tracker,
                        app_embedder=self.ctx.app_embedder,
                        select_top_with_nearest=self.ctx.select_top_with_nearest,
                        extract_features_with_hsv=self.ctx.extract_features_with_hsv,
                        build_fused_appearance_embedding_with_mask=self.ctx.build_fused_appearance_embedding_with_mask,
                    )
                    _, log = update_tracker_from_detections(detections=dets, tracker=self.ctx.tracker, g=float(meta_row.g), reset_mode=bool(meta_row.reset_mode))

                for tid, did in log.get("matches", []):
                    track_rows.append({"frame_idx": frame_idx, "epoch_id": int(log.get("epoch_id", 1)), "local_track_id": int(tid), "det_id": int(did)})
                if self.ctx.save_log:
                    log_rows.append({"frame_idx": frame_idx, "epoch_id": int(log.get("epoch_id", 1)), "g": float(meta_row.g), "reset_mode": bool(meta_row.reset_mode), "log_json": json.dumps(log, default=str)})

                if frame_idx % ck_every == 0:
                    # Flush parquet first, then checkpoint, then manifest.
                    atomic_write_parquet(tracks_out, pd.DataFrame(track_rows, columns=LOCAL_TRACKS_COLUMNS))
                    if self.ctx.save_log:
                        atomic_write_parquet(logs_out, pd.DataFrame(log_rows, columns=TRACKING_LOGS_COLUMNS))
                    save_tracker_checkpoint(ck / f"state_frame_{frame_idx:06d}.pkl", self.ctx.tracker, frame_idx)
                    save_tracker_checkpoint(checkpoint_path, self.ctx.tracker, frame_idx)
                    atomic_write_json(self.manifest_path, {"stage": self.name, "status": "in_progress", "last_flushed_frame": frame_idx, "last_checkpoint_frame": frame_idx, "last_completed_frame": frame_idx, "total_frames": len(frame_ids)})

                self.progress.update("lt", advance=1, description=f"[2/5] Local Tracking batch {batch_no}")

        # Final flush + checkpoint + completed manifest.
        atomic_write_parquet(tracks_out, pd.DataFrame(track_rows, columns=LOCAL_TRACKS_COLUMNS))
        if self.ctx.save_log:
            atomic_write_parquet(logs_out, pd.DataFrame(log_rows, columns=TRACKING_LOGS_COLUMNS))
        final_frame = (max(frame_ids) if frame_ids else 0)
        save_tracker_checkpoint(checkpoint_path, self.ctx.tracker, final_frame)
        atomic_write_json(self.manifest_path, {"stage": self.name, "status": "completed", "last_flushed_frame": final_frame, "last_checkpoint_frame": final_frame, "last_completed_frame": final_frame, "total_frames": len(frame_ids)})


class LocalDetSavingStage(BaseStage):
    name = "local_track_saving"

    def run(self) -> None:
        self.prepare_output()
        matched_dir = self.stage_dir / "matched_dets"
        unmatched_dir = self.stage_dir / "unmatched_dets"
        matched_dir.mkdir(parents=True, exist_ok=True)
        unmatched_dir.mkdir(parents=True, exist_ok=True)

        pre = pd.read_parquet(self.ctx.save_dir / "preprocessed" / "openpose_results.parquet")
        meta = pd.read_parquet(self.ctx.save_dir / "preprocessed" / "frames_metadata.parquet")
        tracks = pd.read_parquet(self.ctx.save_dir / "local_tracking" / "local_tracks.parquet")
        match_idx = {(int(r.frame_idx), int(r.det_id)): int(r.local_track_id) for _, r in tracks.iterrows()}

        frames = sorted(int(v) for v in meta.frame_idx.unique().tolist())
        bs = int(self.ctx.cfg.get("local_det_saving", {}).get("batch_size", 128))

        self.progress.add("lds", "[3/5] Local Det Saving", total=max(len(frames), 1))
        for batch_no, batch in iter_batches(frames, bs):
            for frame_idx in batch:
                mrow = meta.loc[meta.frame_idx == frame_idx].iloc[0]
                img = cv2.imread(str(mrow.frame_path))
                for _, row in pre.loc[pre.frame_idx == frame_idx].iterrows():
                    if pd.isna(row.bbox_x1):
                        continue
                    x1, y1, x2, y2 = [int(row.bbox_x1), int(row.bbox_y1), int(row.bbox_x2), int(row.bbox_y2)]
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size == 0:
                        continue
                    det_id = int(row.det_id)
                    tid = match_idx.get((frame_idx, det_id))
                    out = unmatched_dir / f"frame_{frame_idx:06d}_det_{det_id:03d}.jpg" if tid is None else matched_dir / f"frame_{frame_idx:06d}_track_{tid:03d}_det_{det_id:03d}.jpg"
                    cv2.imwrite(str(out), crop)
                self.progress.update("lds", advance=1, description=f"[3/5] Local Det Saving batch {batch_no}")

        atomic_write_json(self.manifest_path, {"stage": self.name, "status": "completed", "last_completed_frame": (max(frames) if frames else 0), "total_frames": len(frames)})


class GlobalClusteringStage(BaseStage):
    name = "global_clustering"

    def run(self) -> None:
        self.prepare_output()
        checkpoints_dir = self.stage_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        local_ck = self.ctx.save_dir / "local_tracking" / "checkpoints" / "state_latest.pkl"
        if not local_ck.exists():
            raise RuntimeError("Missing local_tracking/checkpoints/state_latest.pkl for global clustering.")

        loaded_tracker, _ = load_tracker_checkpoint(local_ck)
        if not hasattr(loaded_tracker, "get_epoch_tracks"):
            raise RuntimeError("Loaded tracker checkpoint has no get_epoch_tracks()")
        epoch_tracks = loaded_tracker.get_epoch_tracks()

        params = self.ctx.graph_clustering_params or {}
        clusterer = GlobalTrackClusterer(
            k=int(params.get("k", 5)),
            sim_threshold=float(params.get("sim_threshold", 0.5)),
            n_clusters=int(params.get("n_clusters", 2)),
            random_state=int(params.get("random_state", 42)),
            assign_labels=str(params.get("assign_labels", "kmeans")),
        )

        if sum(len(v) for v in epoch_tracks.values()) == 0:
            empty = pd.DataFrame(columns=GLOBAL_MAP_COLUMNS)
            atomic_write_parquet(self.stage_dir / "local_to_global.parquet", empty)
            atomic_write_parquet(self.stage_dir / "global_clusters.parquet", empty)
            np.savez_compressed(str(self.stage_dir / "tracks_similarity.npz"), sim=np.zeros((0, 0), dtype=np.float32), nodes=np.zeros((0, 2), dtype=np.int32))
            atomic_write_json(self.manifest_path, {"stage": self.name, "status": "completed", "last_completed_track": -1, "total_tracks": 0})
            return

        mapping, sim = clusterer.build_mapping(epoch_tracks=epoch_tracks, return_similarity=True)
        rows = [{"epoch_id": int(e), "local_track_id": int(l), "global_track_id": int(g)} for (e, l), g in mapping.items()]

        atomic_write_parquet(self.stage_dir / "local_to_global.parquet", pd.DataFrame(rows, columns=GLOBAL_MAP_COLUMNS))
        atomic_write_parquet(self.stage_dir / "global_clusters.parquet", pd.DataFrame(rows, columns=GLOBAL_MAP_COLUMNS))
        np.savez_compressed(str(self.stage_dir / "tracks_similarity.npz"), sim=sim["sim"], nodes=np.asarray(sim["nodes"], dtype=np.int32))
        save_tracker_checkpoint(checkpoints_dir / "state_latest.pkl", loaded_tracker, len(rows) - 1)
        atomic_write_json(self.manifest_path, {"stage": self.name, "status": "completed", "last_completed_track": len(rows) - 1, "total_tracks": len(rows)})


class GlobalSavingStage(BaseStage):
    name = "global_saving"

    def run(self) -> None:
        self.prepare_output()
        frames_dir = self.stage_dir / "frames"
        videos_dir = self.stage_dir / "videos"
        debug_dir = self.stage_dir / "debug_logs"
        frames_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)
        if self.ctx.save_log:
            debug_dir.mkdir(parents=True, exist_ok=True)

        pre = pd.read_parquet(self.ctx.save_dir / "preprocessed" / "openpose_results.parquet")
        meta = pd.read_parquet(self.ctx.save_dir / "preprocessed" / "frames_metadata.parquet")
        local = pd.read_parquet(self.ctx.save_dir / "local_tracking" / "local_tracks.parquet")
        mapping = pd.read_parquet(self.ctx.save_dir / "global_clustering" / "local_to_global.parquet")
        map_idx = {(int(r.epoch_id), int(r.local_track_id)): int(r.global_track_id) for _, r in mapping.iterrows()}

        manifest = load_manifest(self.manifest_path)
        start = int(manifest.get("last_saved_frame", 0)) + 1 if self.ctx.restore_mode else 1

        frames = sorted(int(v) for v in meta.frame_idx.unique().tolist())
        bs = int(self.ctx.cfg.get("global_saving", {}).get("batch_size", 128))
        self.progress.add("gs", "[5/5] Global Saving", total=max(len(frames), 1))
        if start > 1:
            self.progress.update("gs", completed=start - 1)

        for batch_no, batch in iter_batches([f for f in frames if f >= start], bs):
            for frame_idx in batch:
                mrow = meta.loc[meta.frame_idx == frame_idx].iloc[0]
                img = cv2.imread(str(mrow.frame_path))
                frame_rows = pre.loc[pre.frame_idx == frame_idx]
                detections = []
                for _, row in frame_rows.iterrows():
                    if pd.isna(row.bbox_x1):
                        continue
                    bbox = [float(row.bbox_x1), float(row.bbox_y1), float(row.bbox_x2), float(row.bbox_y2)]
                    kps = np.asarray(row.keypoints, dtype=np.float32) if row.keypoints is not None else None
                    kp_conf = np.asarray(row.kp_conf, dtype=np.float32) if row.kp_conf is not None else None
                    detections.append(_LightDet(bbox, kps, kp_conf))

                matches = []
                for _, mrow in local.loc[local.frame_idx == frame_idx].iterrows():
                    gid = map_idx.get((int(mrow.epoch_id), int(mrow.local_track_id)))
                    if gid is not None:
                        matches.append((int(gid), int(mrow.det_id)))

                render_tracking_overlays(frame=img, detections=detections, matches=matches, frame_idx=frame_idx, use_global_ids=True)
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), img)

                if self.ctx.save_log:
                    (debug_dir / f"frame_{frame_idx:06d}.json").write_text(json.dumps({"frame_idx": frame_idx, "global_matches": matches}), encoding="utf-8")

                atomic_write_json(self.manifest_path, {"stage": self.name, "status": "in_progress", "last_saved_frame": frame_idx, "total_frames": len(frames)})
                self.progress.update("gs", advance=1, description=f"[5/5] Global Saving batch {batch_no}")

        atomic_write_json(self.manifest_path, {"stage": self.name, "status": "completed", "last_saved_frame": (max(frames) if frames else 0), "total_frames": len(frames)})
