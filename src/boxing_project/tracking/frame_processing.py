"""Frame-level tracking helpers shared by inference coordinator and stages.

This module intentionally contains *pure tracking preparation/update* logic and no
stage orchestration, so it can be imported from both `inference_utils.py` and
`tracking_stages.py` without circular imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from boxing_project.tracking.image_utils import (
    attach_overlap_info_to_detections,
    extract_boxing_crops,
    keypoints_to_intersection_bbox,
)
from boxing_project.tracking.tracker import openpose_people_to_detections


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

    This preserves the old `process_frame` behavior by using the same crop and
    overlap enrichment path before tracker update.
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
