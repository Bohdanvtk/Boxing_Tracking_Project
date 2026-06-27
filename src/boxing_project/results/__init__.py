"""Read-only convenience layer over ``dataset/observations.parquet``.

A thin wrapper around :mod:`pandas` and :mod:`numpy` for pulling the ordered
temporal sequence of bbox, BODY_25 keypoints (with confidence) and tracking
metadata for one specific global boxer. It does not touch the inference
pipeline, configs, Docker or packaging, and imports no OpenPose / CUDA / tracker
dependencies.

Typical use::

    from boxing_project.results import BoxingResults

    results = BoxingResults("data/output/test")
    segment = results.global_id(1).epoch(6).window(start_frame=444, length=20)

    model_input = segment.kps             # (20, 25, 3)
    model_mask = segment.detection_mask   # (20,)
"""
from __future__ import annotations

from .dataset import BoxingResults
from .errors import AmbiguousObservationError
from .observation import BBox, FrameObservation
from .segment import BoxerSegment, SegmentCollection, SegmentData
from .selection import TrackSelection

__all__ = [
    "BoxingResults",
    "TrackSelection",
    "BoxerSegment",
    "SegmentCollection",
    "SegmentData",
    "FrameObservation",
    "BBox",
    "AmbiguousObservationError",
]
