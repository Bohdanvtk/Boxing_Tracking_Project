"""Reusable inference helpers for staged tracking.

This module owns non-image pipeline utilities:
- OpenPose initialization
- atomic file/parquet writes
- chunked parquet reads
- DataFrame indexing helpers
- keypoint/confidence normalization for parquet data
- tracker checkpoint/state helpers
- frame-level detection preparation/update logic
- stage-based inference entrypoint
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from boxing_project.tracking.image_utils import (
    attach_overlap_info_to_detections,
    bbox_from_keypoints_for_image,
    clip_bbox_to_image,
    extract_boxing_crops,
    get_detection_bbox,
    keypoints_to_intersection_bbox,
)
from boxing_project.tracking.openpose_processing import set_openpose_module
from boxing_project.tracking.tracker import openpose_people_to_detections


op = None  # set in init_openpose_from_config()

OPENPOSE_RESULTS_COLUMNS = [
    "frame_idx", "frame_path", "det_id", "keypoints", "kp_conf", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "crop_shard_id", "crop_index",
]
FRAMES_METADATA_COLUMNS = ["frame_idx", "frame_path", "g", "reset_mode", "has_detections"]
LOCAL_TRACKS_COLUMNS = ["frame_idx", "epoch_id", "local_track_id", "det_id"]
TRACK_STATES_COLUMNS = [
    "frame_idx", "epoch_id", "local_track_id", "det_id", "is_matched", "confirmed",
    "hits", "age", "time_since_update", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "center_x", "center_y",
]
TRACKING_LOGS_COLUMNS = ["frame_idx", "epoch_id", "g", "reset_mode", "log_json"]
GLOBAL_MAP_COLUMNS = ["epoch_id", "local_track_id", "global_track_id"]


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
        tmp_path.unlink(missing_ok=True)

def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def iter_batches(items: list[int], batch_size: int):
    size = max(1, int(batch_size))
    for i in range(0, len(items), size):
        yield i // size + 1, items[i:i + size]

def chunk_path(chunks_dir: Path, prefix: str, start_frame: int, end_frame: int) -> Path:
    return chunks_dir / f"{prefix}_{start_frame:06d}_{end_frame:06d}.parquet"

def parse_chunk_range(path: Path, prefix: str) -> tuple[int, int] | None:
    m = re.match(rf"^{re.escape(prefix)}_(\d+)_(\d+)\.parquet$", path.name)
    return None if m is None else (int(m.group(1)), int(m.group(2)))

def iter_chunk_files(chunks_dir: Path, prefix: str):
    if not chunks_dir.exists():
        return []
    files = []
    for path in chunks_dir.glob(f"{prefix}_*.parquet"):
        parsed = parse_chunk_range(path, prefix)
        if parsed:
            files.append((*parsed, path))
    return sorted(files, key=lambda x: (x[0], x[1]))

def read_chunked_parquet_for_frames(
    chunks_dir: Path,
    prefix: str,
    frames: list[int],
    columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=expected_columns or columns)

    frame_set = {int(f) for f in frames}
    min_frame, max_frame = min(frame_set), max(frame_set)
    dfs = []

    for start, end, path in iter_chunk_files(chunks_dir, prefix):
        if end < min_frame or start > max_frame:
            continue
        try:
            df = pd.read_parquet(path, columns=columns)
        except (KeyError, ValueError, TypeError):
            df = pd.read_parquet(path)
            if columns is not None:
                df = df[[c for c in columns if c in df.columns]]
        if not df.empty and "frame_idx" in df.columns:
            df = df[df["frame_idx"].isin(frame_set)]
            if not df.empty:
                dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=expected_columns or columns)
    out = pd.concat(dfs, ignore_index=True)
    return out.reindex(columns=expected_columns) if expected_columns else out

def df_by_frame(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if df is None or df.empty or "frame_idx" not in df.columns:
        return {}
    return {int(k): v for k, v in df.groupby("frame_idx", sort=False)}

def first_by_frame(df: pd.DataFrame) -> dict[int, Any]:
    if df is None or df.empty or "frame_idx" not in df.columns:
        return {}
    return {int(r.frame_idx): r for r in df.itertuples(index=False)}

def free(*objects) -> None:
    del objects
    gc.collect()

def normalize_keypoints_xy(value) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        rows = []
        for item in np.asarray(value, dtype=object):
            try:
                item_arr = np.asarray(item, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if item_arr.size >= 2:
                rows.append(item_arr[:2])
        arr = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 2), dtype=np.float32)

    if arr.dtype == object:
        rows = []
        for item in arr:
            try:
                item_arr = np.asarray(item, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if item_arr.size >= 2:
                rows.append(item_arr[:2])
        arr = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 2), dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[: arr.size - arr.size % 2].reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32, copy=False)

def normalize_kp_conf(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    try:
        return np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        vals = []
        for item in np.asarray(value, dtype=object):
            try:
                item_arr = np.asarray(item, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if item_arr.size:
                vals.append(float(item_arr[0]))
        return np.asarray(vals, dtype=np.float32)

def _safe_bool_attr(obj: Any, name: str, default: bool = False) -> bool:
    value = getattr(obj, name, default)
    return bool(value() if callable(value) else value)

def _safe_int_attr(obj: Any, name: str, default: int = 0) -> int:
    value = getattr(obj, name, default)
    return int(value() if callable(value) else value)

def _track_center(track: Any) -> tuple[float | None, float | None]:
    for value in (getattr(track, "last_det_center", None), getattr(track, "center", None)):
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 2 and np.isfinite(arr[:2]).all():
            return float(arr[0]), float(arr[1])
    for name in ("pos", "state"):
        value = getattr(track, name, None)
        if callable(value):
            value = value()
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 2 and np.isfinite(arr[:2]).all():
            return float(arr[0]), float(arr[1])
    return None, None

def _bbox_from_detection(det: Any, img_shape) -> list[float] | None:
    # No keypoint fallback here: matched tracks should use the bbox stored on the Detection.
    return clip_bbox_to_image(get_detection_bbox(det), img_shape)

def _track_bbox_from_track(track: Any, img_shape) -> list[float] | None:
    for attr in ("last_bbox", "bbox", "tlbr", "last_tlbr"):
        value = getattr(track, attr, None)
        value = value() if callable(value) else value
        bbox = clip_bbox_to_image(value, img_shape)
        if bbox is not None:
            return bbox
    kps = getattr(track, "last_keypoints", None)
    if kps is None:
        return None
    return bbox_from_keypoints_for_image(normalize_keypoints_xy(kps), normalize_kp_conf(getattr(track, "last_kp_conf", None)), img_shape)

def collect_tracker_state_rows(*, tracker, frame_idx: int, epoch_id: int, matches, img_shape, detections=None):
    matched_det_by_track = {int(tid): int(did) for tid, did in (matches or [])}
    detections = detections or []
    rows = []
    for trk in getattr(tracker, "tracks", []) or []:
        tid = int(getattr(trk, "track_id"))
        det_id = matched_det_by_track.get(tid)
        bbox = _bbox_from_detection(detections[det_id], img_shape) if det_id is not None and 0 <= det_id < len(detections) else None
        bbox = bbox or _track_bbox_from_track(trk, img_shape)
        cx, cy = _track_center(trk)
        rows.append({
            "frame_idx": int(frame_idx), "epoch_id": int(epoch_id), "local_track_id": tid,
            "det_id": int(det_id) if det_id is not None else -1, "is_matched": det_id is not None,
            "confirmed": _safe_bool_attr(trk, "confirmed", False), "hits": _safe_int_attr(trk, "hits", 0),
            "age": _safe_int_attr(trk, "age", 0), "time_since_update": _safe_int_attr(trk, "time_since_update", 0),
            "bbox_x1": None if bbox is None else float(bbox[0]), "bbox_y1": None if bbox is None else float(bbox[1]),
            "bbox_x2": None if bbox is None else float(bbox[2]), "bbox_y2": None if bbox is None else float(bbox[3]),
            "center_x": None if cx is None else float(cx), "center_y": None if cy is None else float(cy),
        })
    return rows

def save_tracker_checkpoint(path: Path, tracker: Any, last_completed_frame: int) -> None:
    payload = {"mode": "pickle_tracker", "tracker": tracker, "last_completed_frame": int(last_completed_frame)}
    try:
        atomic_write_bytes(path, pickle.dumps(payload))
    except (pickle.PickleError, TypeError, AttributeError) as exc:
        raise RuntimeError("Failed to serialize tracker checkpoint; implement explicit tracker state serialization.") from exc

def load_tracker_checkpoint(path: Path) -> tuple[Any, int]:
    try:
        payload = pickle.loads(path.read_bytes())
    except (pickle.PickleError, EOFError, AttributeError, ImportError, IndexError) as exc:
        raise RuntimeError(f"Corrupt tracker checkpoint: {path}") from exc
    if payload.get("mode") != "pickle_tracker":
        raise RuntimeError(f"Unsupported checkpoint mode: {payload.get('mode')}")
    return payload["tracker"], int(payload.get("last_completed_frame", -1))

def top_track_ids_by_hits(tracker, k: int = 4) -> set[int]:
    """Return up to `k` active track ids sorted by descending `hits`.

    The adaptive detector-selection logic uses this to increase candidate count
    when the strongest tracks are suddenly missed.
    """
    tracks = tracker.get_active_tracks(confirmed_only=False)
    tracks = sorted(tracks, key=lambda t: int(getattr(t, "hits", 0)), reverse=True)
    return {int(t.track_id) for t in tracks[:k]}

def attach_center_distance_overlap_metadata(detections, cfg) -> None:
    """Attach adaptive center-distance overlap risk metadata to detections."""
    centers = [np.asarray(det.center, dtype=np.float32) for det in detections]
    if len(centers) < 2:
        return
    img_diag = float(max(getattr(cfg, "image_diag", 1.0), 1.0))
    threshold = float(getattr(cfg, "center_dist_overlap_threshold", 0.08))
    for i, det_i in enumerate(detections):
        best = 1e9
        for j, det_j in enumerate(detections):
            if i == j:
                continue
            d = float(np.linalg.norm(centers[i] - centers[j])) / img_diag
            best = min(best, d)
        det_i.meta["center_dist_norm_min"] = None if best == 1e9 else best
        det_i.meta["center_dist_overlap_risk"] = bool(best <= threshold) if best != 1e9 else False

def prepare_frame_detections_from_keypoints(
    *,
    kps: np.ndarray,
    original_img,
    conf_th: float,
    tracker,
    app_embedder,
    select_top_with_nearest,
    extract_features_with_hsv,
    build_fused_appearance_embedding_with_mask,
):
    """Convert raw OpenPose keypoints to tracker detections with appearance data.

    This builds detections with crop features and overlap metadata before tracker update.
    """
    extra_n = int(getattr(tracker, "_adaptive_extra_n", 9))
    kps = select_top_with_nearest(kps, conf_th=conf_th, top_count=3, n=extra_n, intersect=2)

    h, w = original_img.shape[:2]
    people = []
    for person_kps in kps:
        intersection_bb = keypoints_to_intersection_bbox(
            person_kps,
            conf_th=conf_th,
            img_w=w,
            img_h=h,
        )
        parts = extract_boxing_crops(frame_bgr=original_img, kps=person_kps, conf_threshold=conf_th)
        people.append(
            {
                "keypoints": person_kps,
                "bbox": intersection_bb,
                "bbox_for_intersection": intersection_bb,
                "left_glove_crop": parts["left_glove"],
                "right_glove_crop": parts["right_glove"],
                "shorts_crop": parts["shorts"],
            }
        )

    detections = openpose_people_to_detections(people, min_kp_conf=tracker.cfg.min_kp_conf)

    attach_overlap_info_to_detections(
        detections=detections,
        overlap_threshold=tracker.cfg.overlap_log_threshold,
        skeleton_overlap_threshold=tracker.cfg.skeleton_overlap_threshold,
        skeleton_overlap_full_weight=tracker.cfg.skeleton_overlap_full_weight,
        skeleton_overlap_core_weight=tracker.cfg.skeleton_overlap_core_weight,
        skeleton_overlap_conf_threshold=tracker.cfg.skeleton_overlap_conf_threshold,
        skeleton_overlap_thickness=tracker.cfg.skeleton_overlap_thickness,
    )

    attach_center_distance_overlap_metadata(detections, tracker.cfg)

    for det in detections:
        raw = det.meta.get("raw", {})
        bbox = raw.get("bbox", None)

        left_glove_crop = raw.get("left_glove_crop")
        right_glove_crop = raw.get("right_glove_crop")
        shorts_crop = raw.get("shorts_crop")

        body_feat = app_embedder.embed(original_img, bbox) if (app_embedder is not None and bbox is not None) else None
        left_glove_features = extract_features_with_hsv(left_glove_crop)
        right_glove_features = extract_features_with_hsv(right_glove_crop)
        shorts_features = extract_features_with_hsv(shorts_crop)

        det.meta["body_features"] = body_feat
        det.meta["left_glove_features"] = left_glove_features
        det.meta["right_glove_features"] = right_glove_features
        det.meta["shorts_features"] = shorts_features

        e_app, e_app_valid_mask, e_app_coverage = build_fused_appearance_embedding_with_mask(
            body_feat,
            left_glove_features,
            right_glove_features,
            shorts_features,
            w_body=getattr(tracker.cfg, "w_body", 1.0),
            w_left_glove=getattr(tracker.cfg, "w_left_glove", 0.5),
            w_right_glove=getattr(tracker.cfg, "w_right_glove", 0.5),
            w_shorts=getattr(tracker.cfg, "w_shorts", 0.75),
        )
        det.meta["e_app"] = e_app
        det.meta["e_app_valid_mask"] = e_app_valid_mask
        det.meta["e_app_coverage"] = e_app_coverage

    return detections

def update_tracker_from_detections(*, detections, tracker, g: float, reset_mode: bool):
    """Update tracker and preserve adaptive candidate-count behavior."""
    top_before = top_track_ids_by_hits(tracker, k=4)
    extra_n = int(getattr(tracker, "_adaptive_extra_n", 9))

    log = tracker.update(detections, g=g, reset_mode=reset_mode)

    matched_track_ids = {int(track_id) for track_id, _ in log.get("matches", [])}
    missed_top_track_ids = top_before - matched_track_ids

    tracker._adaptive_extra_n = extra_n + 9 if missed_top_track_ids else 7
    log["adaptive_extra_n_used"] = int(extra_n)
    log["adaptive_extra_n_next"] = int(tracker._adaptive_extra_n)
    log["missed_top_track_ids"] = sorted(missed_top_track_ids)

    return detections, log

def init_openpose_from_config(openpose_cfg: dict):
    """Initialize OpenPose wrapper from YAML config and return (op_module, opWrapper)."""
    global op

    root = os.path.expanduser(openpose_cfg["root"])
    py_path = os.path.join(root, "build", "python")

    if py_path not in sys.path:
        sys.path.append(py_path)

    try:
        from openpose import pyopenpose as op_module
    except Exception as e:
        raise RuntimeError(
            "Failed to import pyopenpose. Check OpenPose installation and the 'root' path."
        ) from e

    op = op_module
    set_openpose_module(op_module)

    params = {
        "model_folder": os.path.join(root, "models"),
        "hand": openpose_cfg.get("hand", False),
        "face": openpose_cfg.get("face", False),
        "net_resolution": openpose_cfg.get("net_resolution", "-1x256"),
        "num_gpu": openpose_cfg.get("num_gpu", 1),
        "num_gpu_start": openpose_cfg.get("num_gpu_start", 0),
        "render_pose": openpose_cfg.get("render_pose", 0),
        "disable_blending": openpose_cfg.get("disable_blending", True),
        "number_people_max": openpose_cfg.get("number_people_max", 5),
        "disable_multi_thread": openpose_cfg.get("disable_multi_thread", True),
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    return op_module, opWrapper

def extract_features_with_hsv(image: np.ndarray) -> np.ndarray | None:
    """Extract a 32-dimensional HSV color histogram feature vector."""
    if image is None or image.size == 0:
        return None

    image = cv2.resize(image, (32, 32))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist.astype(np.float32)

def _l2_normalize_feature(vec):
    if vec is None:
        return None

    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None

    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None

    return (arr / norm).astype(np.float32)

def build_fused_appearance_embedding_with_mask(
    body_feat,
    left_glove_feat,
    right_glove_feat,
    shorts_feat,
    *,
    w_body: float = 1.0,
    w_left_glove: float = 0.5,
    w_right_glove: float = 0.5,
    w_shorts: float = 0.75,
    part_dim: int = 32,
):
    """Build fixed-length fused appearance vector plus valid-part mask and coverage."""
    body = _l2_normalize_feature(body_feat)
    if body is None:
        return None, None, 0.0

    part_dim = max(0, int(part_dim))
    weights = {
        "body": float(w_body),
        "left_glove": float(w_left_glove),
        "right_glove": float(w_right_glove),
        "shorts": float(w_shorts),
    }
    total_weight = sum(max(weight, 0.0) for weight in weights.values())
    visible_weight = max(weights["body"], 0.0)

    def _part_or_zero_and_mask(part_feat, weight):
        nonlocal visible_weight

        part = _l2_normalize_feature(part_feat)
        if part is None or part.size != part_dim:
            return np.zeros(part_dim, dtype=np.float32), np.zeros(part_dim, dtype=bool)

        visible_weight += max(float(weight), 0.0)
        return part, np.ones(part_dim, dtype=bool)

    left, left_mask = _part_or_zero_and_mask(left_glove_feat, weights["left_glove"])
    right, right_mask = _part_or_zero_and_mask(right_glove_feat, weights["right_glove"])
    shorts, shorts_mask = _part_or_zero_and_mask(shorts_feat, weights["shorts"])

    fused = np.concatenate(
        [
            weights["body"] * body,
            weights["left_glove"] * left,
            weights["right_glove"] * right,
            weights["shorts"] * shorts,
        ],
        axis=0,
    ).astype(np.float32)

    e_app = _l2_normalize_feature(fused)
    if e_app is None:
        return None, None, 0.0

    e_app_valid_mask = np.concatenate(
        [
            np.ones(body.size, dtype=bool),
            left_mask,
            right_mask,
            shorts_mask,
        ],
        axis=0,
    )
    e_app_coverage = 0.0 if total_weight <= 1e-12 else visible_weight / total_weight
    e_app_coverage = float(np.clip(e_app_coverage, 0.0, 1.0))

    return e_app, e_app_valid_mask, e_app_coverage

def _kp_center(kp: np.ndarray, conf_th: float):
    valid = kp[:, 2] > conf_th
    if not np.any(valid):
        return None
    return np.mean(kp[valid, :2], axis=0)

def _center_dist(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.linalg.norm(a - b))

def select_top_with_nearest(
    kps: np.ndarray,
    conf_th: float,
    top_count: int = 3,
    n: int = 3,
    intersect: int = 1,
) -> np.ndarray:
    """Keep top detections by OpenPose order, then add up to n nearest extra detections."""
    total = len(kps)
    if total <= top_count:
        return kps

    top_count = min(top_count, total)
    top_indices = list(range(top_count))
    extra_indices = list(range(top_count, total))

    centers = [_kp_center(kps[i], conf_th) for i in range(total)]
    usage_count = {idx: 0 for idx in extra_indices}
    selected = list(top_indices)
    selected_extras = []

    pairs = []
    for top_idx in top_indices:
        for extra_idx in extra_indices:
            dist = _center_dist(centers[top_idx], centers[extra_idx])
            pairs.append((dist, extra_idx))

    for _, extra_idx in sorted(pairs, key=lambda x: x[0]):
        if len(selected_extras) >= n:
            break
        if usage_count[extra_idx] >= intersect:
            continue

        if extra_idx not in selected:
            selected.append(extra_idx)
            selected_extras.append(extra_idx)

        usage_count[extra_idx] += 1

    return kps[selected]

def visualize_sequence(
    opWrapper,
    tracker,
    app_emb_path,
    sb_cfg: dict,
    images,
    save_width,
    save_dir: Path | None,
    graph_clustering_params: dict | None = None,
    pipeline_cfg: dict | None = None,
    progress=None,
):
    from boxing_project.apperance_embedding.inference import AppearanceEmbedder, AppearanceEmbedConfig
    from boxing_project.tracking.progress_utils import RichStageProgress
    from boxing_project.tracking.tracking_stages import (
        PipelineContext,
        PreprocessingStage,
        LocalTrackingStage,
        LocalDetSavingStage,
        GlobalClusteringStage,
        GlobalSavingStage,
    )

    runtime_cfg = pipeline_cfg or {}
    cfg = {
        "stages": runtime_cfg.get(
            "stages",
            {
                "preprocessing": True,
                "local_tracking": True,
                "local_det_saving": True,
                "global_clustering": True,
                "global_saving": True,
            },
        ),
        "restore_mode": bool(runtime_cfg.get("restore_mode", False)),
        "save_log": bool(runtime_cfg.get("save_log", False)),
        "preprocessing": runtime_cfg.get("preprocessing", {}),
        "local_tracking": runtime_cfg.get("local_tracking", {}),
        "local_det_saving": runtime_cfg.get("local_det_saving", {}),
        "global_clustering": runtime_cfg.get("global_clustering", {}),
        "global_saving": runtime_cfg.get("global_saving", {}),
        "progress": runtime_cfg.get("progress", {"enabled": True, "library": "rich"}),
    }

    owns_progress = progress is None
    if progress is None:
        progress = RichStageProgress(enabled=bool(cfg.get("progress", {}).get("enabled", True)))

    try:
        progress.update("setup", description="[0/5] Initializing appearance embedder")
        app_embedder = AppearanceEmbedder(AppearanceEmbedConfig(model_path=app_emb_path))

        save_dir = Path(save_dir) if save_dir is not None else Path("output")
        ctx = PipelineContext(
            opWrapper=opWrapper,
            tracker=tracker,
            app_embedder=app_embedder,
            sb_cfg=sb_cfg,
            images=list(images),
            save_width=save_width,
            save_dir=save_dir,
            save_log=cfg["save_log"],
            restore_mode=cfg["restore_mode"],
            cfg=cfg,
            graph_clustering_params=graph_clustering_params or {},
            select_top_with_nearest=select_top_with_nearest,
            extract_features_with_hsv=extract_features_with_hsv,
            build_fused_appearance_embedding_with_mask=build_fused_appearance_embedding_with_mask,
        )

        run_local_det_saving = bool(cfg["stages"].get("local_det_saving", True))
        run_global_clustering = (
            bool(cfg["stages"].get("global_clustering", True))
            and bool(cfg.get("global_clustering", {}).get("enabled", True))
        )
        run_global_saving = (
            run_global_clustering
            and bool(cfg["stages"].get("global_saving", True))
            and bool(cfg.get("global_saving", {}).get("enabled", True))
        )

        PreprocessingStage(ctx, progress).run()
        LocalTrackingStage(ctx, progress).run()

        if run_local_det_saving:
            LocalDetSavingStage(ctx, progress).run()

        if run_global_clustering:
            GlobalClusteringStage(ctx, progress).run()
            if run_global_saving:
                GlobalSavingStage(ctx, progress).run()

    finally:
        if owns_progress:
            progress.finish()