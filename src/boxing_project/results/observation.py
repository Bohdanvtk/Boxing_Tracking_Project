"""Single-frame primitives: :class:`FrameObservation` and :class:`BBox`."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import BBOX_COLUMNS, META_COLUMNS
from .utils import _is_missing, empty_keypoints, row_keypoints


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box ``[x1, y1, x2, y2]``.

    Convertible to a ``(4,)`` numpy array via ``np.asarray(bbox)`` or
    :meth:`to_numpy`.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=float)

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self.to_numpy()
        return arr.astype(dtype) if dtype is not None else arr

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class FrameObservation:
    """One boxer on one frame.

    Three states are possible:

    * **detected** -- a parquet row with a matched detection and real keypoints
      (``is_observed`` and ``has_detection`` both True);
    * **observation without detection** -- a parquet row exists (the track was
      active) but no detection was matched, so the bbox is the track/Kalman
      state and the keypoints are NaN/0.0 (``is_observed`` True,
      ``has_detection`` False);
    * **padded** -- no parquet row at all, created only to fill a fixed-length
      :meth:`window` position (``is_observed`` False).
    """

    def __init__(self, frame_idx: int, global_track_id, row: pd.Series | None):
        self._frame_idx = int(frame_idx)
        self._global_track_id = global_track_id
        self._row = row
        self._kps: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"FrameObservation(frame_idx={self._frame_idx}, "
            f"global_track_id={self._global_track_id}, observed={self.is_observed})"
        )

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def is_observed(self) -> bool:
        """True when a real parquet row backs this observation."""
        return self._row is not None

    @property
    def has_keypoints(self) -> bool:
        """True when this row carries a real BODY_25 keypoints payload."""
        if self._row is None:
            return False
        return not _is_missing(self._row.get("keypoints"))

    @property
    def has_detection(self) -> bool:
        """True when this row carries a matched detection with real keypoints.

        The Stage 7 producer fills ``keypoints``/``kp_conf`` only from a matched
        OpenPose detection, so in the current dataset ``is_matched`` and the
        presence of keypoints coincide. This property requires both, and is what
        :attr:`BoxerSegment.detection_mask` is built from. Use
        :attr:`has_keypoints` if you only care about keypoint availability.
        """
        if self._row is None:
            return False
        matched = self._row["is_matched"] if "is_matched" in self._row else None
        matched_ok = bool(matched) if not _is_missing(matched) else False
        return matched_ok and self.has_keypoints

    @property
    def bbox(self) -> BBox:
        """Box for this frame. NaN box when this is a padded position."""
        if self._row is None:
            return BBox(math.nan, math.nan, math.nan, math.nan)
        values = [float(self._row[column]) for column in BBOX_COLUMNS]
        return BBox(*values)

    @property
    def kps(self) -> np.ndarray:
        """``(25, 3)`` array of ``[x, y, confidence]``."""
        if self._kps is None:
            self._kps = empty_keypoints() if self._row is None else row_keypoints(self._row)
        return self._kps

    @property
    def meta(self) -> dict:
        """Tracking metadata for this frame plus ``n_visible_keypoints``.

        For a padded position only ``frame_idx`` and ``global_track_id`` are
        populated; metadata is never carried over from a neighbouring frame.
        """
        kps = self.kps
        n_visible = int(np.count_nonzero(kps[:, 2] > 0))
        if self._row is None:
            meta = {column: None for column in META_COLUMNS}
            meta["frame_idx"] = self._frame_idx
            meta["global_track_id"] = self._global_track_id
        else:
            meta = {
                column: (self._row[column] if column in self._row else None)
                for column in META_COLUMNS
            }
        meta["n_visible_keypoints"] = n_visible
        return meta
