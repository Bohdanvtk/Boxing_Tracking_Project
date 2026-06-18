"""Helpers for Stage 7 (``DatasetExportStage``).

Builds the public dataset file::

    <save_dir>/dataset/observations.parquet

One row represents one active local track on one frame, including frames where
no detection was matched.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from boxing_project.tracking.inference_utils import iter_chunk_files


OBSERVATIONS_COLUMNS = [
    # identity
    "frame_idx", "epoch_id", "local_track_id", "global_track_id",
    # track state
    "det_id", "is_matched", "confirmed", "hits", "age", "time_since_update",
    # authoritative geometry from track_states
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "center_x", "center_y",
    # matched detection payload
    "keypoints", "kp_conf",
    # overlap metadata
    "is_overlapping", "max_overlap_iou", "min_center_dist_norm",
    "center_dist_norm_det_idx",
    # appearance availability / coverage
    "has_body_features", "has_left_glove_features", "has_right_glove_features",
    "has_shorts_features", "has_e_app", "e_app_coverage",
]

# Detection data only; geometry stays authoritative in track_states.
PREPARED_KEEP_COLUMNS = [
    "frame_idx", "det_id", "keypoints", "kp_conf",
    "is_overlapping", "max_overlap_iou", "min_center_dist_norm",
    "center_dist_norm_det_idx",
    "has_body_features", "has_left_glove_features", "has_right_glove_features",
    "has_shorts_features", "has_e_app", "e_app_coverage",
]

# Missing values must be None to preserve Arrow list-or-null types.
_LIST_COLUMNS = ("keypoints", "kp_conf")


def read_all_chunks(
    chunks_dir: Path,
    prefix: str,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Read and concatenate all matching chunks.

    Loading all chunks into memory is intentional because this runs only in the
    final export stage. Returns an empty frame with the expected schema when no
    chunks exist.
    """
    frames = [
        pd.read_parquet(path)
        for _, _, path in iter_chunk_files(chunks_dir, prefix)
    ]

    if not frames:
        return pd.DataFrame(columns=expected_columns)

    df = pd.concat(frames, ignore_index=True)
    return df.reindex(
        columns=[column for column in expected_columns if column in df.columns]
    )


def build_observations(
    track_states: pd.DataFrame,
    prepared: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Build one observation per active local track per frame.

    LEFT joins preserve unmatched and unmapped tracks. Export-specific null
    normalization is applied before enforcing the final schema.
    """
    dup_det = prepared.duplicated(subset=["frame_idx", "det_id"])
    if dup_det.any():
        raise RuntimeError(
            f"Duplicate (frame_idx, det_id) keys in prepared_detections: {int(dup_det.sum())} rows"
        )

    dup_map = mapping.duplicated(subset=["epoch_id", "local_track_id"])
    if dup_map.any():
        raise RuntimeError(
            f"Duplicate (epoch_id, local_track_id) keys in local_to_global: {int(dup_map.sum())} rows"
        )

    det = prepared[
        [column for column in PREPARED_KEEP_COLUMNS if column in prepared.columns]
    ]

    obs = track_states.merge(
        det,
        on=["frame_idx", "det_id"],
        how="left",
        validate="many_to_one",
    )
    obs = obs.merge(
        mapping,
        on=["epoch_id", "local_track_id"],
        how="left",
        validate="many_to_one",
    )

    # Convert the unmatched sentinel to a nullable integer value.
    if "det_id" in obs.columns:
        obs["det_id"] = obs["det_id"].astype("Int64")
        obs.loc[obs["det_id"] == -1, "det_id"] = pd.NA

    for column in _LIST_COLUMNS:
        if column in obs.columns:
            obs[column] = obs[column].where(obs[column].notna(), None)

    for column in OBSERVATIONS_COLUMNS:
        if column not in obs.columns:
            obs[column] = None

    return (
        obs[OBSERVATIONS_COLUMNS]
        .sort_values(
            [
                "global_track_id",
                "epoch_id",
                "local_track_id",
                "frame_idx",
            ]
        )
        .reset_index(drop=True)
    )