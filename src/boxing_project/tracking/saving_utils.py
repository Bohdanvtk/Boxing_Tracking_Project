from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _clip_bbox_xyxy(bbox, w: int, h: int):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _save_matched_det(
    *,
    frame_dir: Path,
    track_id: int,
    frame: np.ndarray,
    bbox,
    keypoints: np.ndarray | None,
    kp_conf: np.ndarray | None,
    conf_th: float,
) -> None:
    h, w = frame.shape[:2]
    track_dir = frame_dir / f"track_{track_id}"
    track_dir.mkdir(parents=True, exist_ok=True)

    bb = _clip_bbox_xyxy(bbox, w=w, h=h)
    if bb is not None:
        x1, y1, x2, y2 = bb
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(str(track_dir / "crop.jpg"), crop)

    if keypoints is None or kp_conf is None:
        kps4 = np.zeros((0, 4), dtype=np.float32)
    else:
        xy = keypoints.astype(np.float32, copy=False)
        conf = kp_conf.astype(np.float32, copy=False)

        if xy.ndim != 2 or xy.shape[1] != 2:
            xy = xy.reshape((-1, 2)).astype(np.float32, copy=False)

        conf = conf.reshape((-1,))
        k = min(xy.shape[0], conf.shape[0])
        xy = xy[:k]
        conf = conf[:k]

        finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        mask = (conf >= float(conf_th)) & finite

        kps4 = np.concatenate([xy, conf[:, None], mask.astype(np.float32)[:, None]], axis=1).astype(np.float32, copy=False)
        kps4[:, :2] = np.nan_to_num(kps4[:, :2], nan=0.0, posinf=0.0, neginf=0.0)

    np.savez_compressed(str(track_dir / "kps.npz"), kps=kps4)


def _save_frame_extra(*, frame_dir: Path, unprocessed_frame: np.ndarray, detections) -> None:
    h, w = unprocessed_frame.shape[:2]
    extra_dir = frame_dir / "extra"
    extra_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(extra_dir / "unprocessed_image.jpg"), unprocessed_frame)

    for det_idx, det in enumerate(detections):
        raw = det.meta.get("raw", {})
        bb = _clip_bbox_xyxy(raw.get("bbox", None), w=w, h=h)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        crop = unprocessed_frame[y1:y2, x1:x2]
        cv2.imwrite(str(extra_dir / f"det_{det_idx:03d}.jpg"), crop)




def _save_frame_debug_txt(*, debug_dir: Path, frame_idx: int, frame_log) -> None:
    """Save textual matrix debug from tracking_debug.DebugLog buffer."""
    if frame_log is None or not hasattr(frame_log, "buffer"):
        return

    lines = list(getattr(frame_log, "buffer", []) or [])
    if not lines:
        return

    txt = "\n".join([f"FRAME {int(frame_idx):06d}", "=" * 80, "", *lines, "", "=" * 80])
    (debug_dir / "tracking_debug.txt").write_text(txt, encoding="utf-8")


def _save_frame_debug(*, frame_dir: Path, detections, tracker, log: dict) -> None:
    debug_dir = frame_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    matches = {int(det_idx): int(track_id) for track_id, det_idx in log.get("matches", [])}
    tracks_by_id = {int(t.track_id): t for t in tracker.tracks}

    _save_frame_debug_txt(
        debug_dir=debug_dir,
        frame_idx=int(log.get("frame_idx", -1)),
        frame_log=log.get("frame_log", None),
    )

    for det_idx, det in enumerate(detections):
        raw = det.meta.get("raw", {}) if isinstance(det.meta, dict) else {}
        bbox = raw.get("bbox", None)
        track_id = matches.get(det_idx)
        trk = tracks_by_id.get(track_id) if track_id is not None else None

        rec: dict[str, Any] = {
            "frame_idx": int(log.get("frame_idx", -1)),
            "det_idx": int(det_idx),
            "bbox_xyxy": list(map(float, bbox)) if bbox is not None else None,
            "det_center": list(map(float, det.center)) if det.center is not None else None,
            "has_e_app": bool(det.meta.get("e_app") is not None),
            "e_app_error": det.meta.get("e_app_error", None) if isinstance(det.meta, dict) else None,
            "matched_track_id": int(track_id) if track_id is not None else None,
            "track": None,
        }

        if trk is not None:
            rec["track"] = {
                "track_id": int(trk.track_id),
                "confirmed": bool(trk.confirmed),
                "age": int(trk.age),
                "hits": int(trk.hits),
                "time_since_update": int(trk.time_since_update),
                "post_reset_mode": bool(trk.post_reset_mode),
                "post_reset_age": int(trk.post_reset_age),
                "bad_kp_streak": int(trk.bad_kp_streak),
                "center": [float(x) for x in trk.pos()],
                "state": np.asarray(trk.state, dtype=float).tolist(),
                "last_det_center": [float(x) for x in trk.last_det_center] if trk.last_det_center is not None else None,
                "previous_kps": None if trk.last_keypoints is None else np.asarray(trk.last_keypoints, dtype=float).tolist(),
                "previous_kp_conf": None if trk.last_kp_conf is None else np.asarray(trk.last_kp_conf, dtype=float).tolist(),
            }

        (debug_dir / f"det_{det_idx:03d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def save_tracking_outputs(
    *,
    save_dir: Path,
    frame_idx: int,
    original_frame: np.ndarray,
    processed_frame: np.ndarray,
    detections,
    log: dict,
    conf_th: float,
    tracker,
) -> None:
    frame_dir = Path(save_dir) / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    vis_path = frame_dir / "frame_vis.jpg"
    if processed_frame is not None and not vis_path.exists():
        cv2.imwrite(str(vis_path), processed_frame)

    if bool(tracker.cfg.save_log):
        from boxing_project.tracking.tracking_debug import GENERAL_LOG

        (Path(save_dir) / "debug_log.txt").write_text("\n".join(GENERAL_LOG), encoding="utf-8")

    _save_frame_extra(frame_dir=frame_dir, unprocessed_frame=original_frame, detections=detections)

    if bool(tracker.cfg.debug):
        debug_log = dict(log)
        debug_log["frame_idx"] = int(frame_idx)
        _save_frame_debug(frame_dir=frame_dir, detections=detections, tracker=tracker, log=debug_log)

    for track_id, det_idx in log.get("matches", []):
        if det_idx < 0 or det_idx >= len(detections):
            continue

        det = detections[det_idx]
        raw = det.meta.get("raw", {})
        _save_matched_det(
            frame_dir=frame_dir,
            track_id=int(track_id),
            frame=original_frame,
            bbox=raw.get("bbox", None),
            keypoints=det.keypoints,
            kp_conf=det.kp_conf,
            conf_th=conf_th,
        )
