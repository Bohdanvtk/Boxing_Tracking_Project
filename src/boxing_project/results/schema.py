"""Schema constants for ``observations.parquet``.

These names mirror the columns written by Stage 7
(:mod:`boxing_project.utils.dataset_export`). The results package never writes
parquet; these constants only describe what it reads.
"""
from __future__ import annotations

# BODY_25 keypoint count.
N_KEYPOINTS = 25

# bbox is stored as four scalar columns (authoritative track geometry).
BBOX_COLUMNS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

# Tracking metadata exposed on ``FrameObservation.meta`` / ``BoxerSegment.meta``.
# ``n_visible_keypoints`` is computed from ``kp_conf`` and appended separately.
META_COLUMNS = [
    "frame_idx",
    "epoch_id",
    "local_track_id",
    "global_track_id",
    "det_id",
    "is_matched",
    "confirmed",
    "hits",
    "age",
    "time_since_update",
    "is_overlapping",
    "max_overlap_iou",
    "min_center_dist_norm",
    "center_dist_norm_det_idx",
    "has_body_features",
    "has_left_glove_features",
    "has_right_glove_features",
    "has_shorts_features",
    "has_e_app",
    "e_app_coverage",
]

# Minimum columns the API depends on. A parquet missing any of these is rejected.
REQUIRED_COLUMNS = [
    "frame_idx",
    "epoch_id",
    "local_track_id",
    "global_track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "keypoints",
    "kp_conf",
    "is_matched",
    "confirmed",
]

# Conventional file name inside an output directory's ``dataset/`` folder.
DEFAULT_PARQUET_NAME = "observations.parquet"
