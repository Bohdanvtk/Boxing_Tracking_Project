"""Per-boxer temporal sequences.

:class:`BoxerSegment` is one global id across an ordered sequence of frames;
:class:`SegmentCollection` groups several of them by global id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .errors import AmbiguousObservationError
from .observation import FrameObservation
from .schema import META_COLUMNS, N_KEYPOINTS
from .utils import _as_id


@dataclass(frozen=True)
class SegmentData:
    """Plain container returned by :attr:`BoxerSegment.data`."""

    frames: np.ndarray
    bbox: np.ndarray
    kps: np.ndarray
    meta: pd.DataFrame
    observation_mask: np.ndarray
    detection_mask: np.ndarray


class BoxerSegment:
    """One global id across an ordered sequence of frames.

    The segment is always ordered by ``frame_idx``. A single global id may be
    backed by several ``(epoch_id, local_track_id)`` fragments; per-frame
    metadata keeps the real ``epoch_id``/``local_track_id``.

    When built with an explicit ``frame_axis`` (e.g. :meth:`window`) every
    requested frame becomes a time position, padded when no row exists for it.
    """

    def __init__(
        self,
        global_track_id,
        df: pd.DataFrame,
        frame_axis: Sequence[int] | None = None,
    ):
        self._gid = _as_id(global_track_id)
        self._df = df.sort_values("frame_idx").reset_index(drop=True)
        if frame_axis is None:
            self._frame_axis = [int(f) for f in sorted(self._df["frame_idx"].unique())]
        else:
            self._frame_axis = [int(f) for f in frame_axis]
        self._observations: list[FrameObservation] | None = None

    def _resolve(self) -> list[FrameObservation]:
        if self._observations is not None:
            return self._observations

        by_frame = {int(f): group for f, group in self._df.groupby("frame_idx")}
        observations: list[FrameObservation] = []
        for frame in self._frame_axis:
            group = by_frame.get(frame)
            if group is None or len(group) == 0:
                observations.append(FrameObservation(frame, self._gid, None))
            elif len(group) == 1:
                observations.append(FrameObservation(frame, self._gid, group.iloc[0]))
            else:
                raise AmbiguousObservationError(self._gid, frame, group)
        self._observations = observations
        return observations

    # -- identity ----------------------------------------------------------
    @property
    def global_id(self):
        return self._gid

    @property
    def df(self) -> pd.DataFrame:
        """Raw backing rows (present rows only).

        Mutating this DataFrame may affect the segment's internal state; use
        :meth:`to_pandas` for an independent copy.
        """
        return self._df

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        """Return the backing rows; an independent copy by default."""
        return self._df.copy() if copy else self._df

    def __len__(self) -> int:
        return len(self._frame_axis)

    def __iter__(self) -> Iterator[FrameObservation]:
        return iter(self._resolve())

    def __repr__(self) -> str:
        return f"BoxerSegment(global_id={self._gid}, T={len(self)})"

    # -- arrays ------------------------------------------------------------
    @property
    def frames(self) -> np.ndarray:
        return np.asarray(self._frame_axis, dtype=int)

    @property
    def bbox(self) -> np.ndarray:
        observations = self._resolve()
        if not observations:
            return np.empty((0, 4), dtype=float)
        return np.stack([obs.bbox.to_numpy() for obs in observations])

    @property
    def kps(self) -> np.ndarray:
        observations = self._resolve()
        if not observations:
            return np.empty((0, N_KEYPOINTS, 3), dtype=float)
        return np.stack([obs.kps for obs in observations])

    @property
    def meta(self) -> pd.DataFrame:
        observations = self._resolve()
        columns = META_COLUMNS + ["n_visible_keypoints"]
        rows = [obs.meta for obs in observations]
        return pd.DataFrame(rows, columns=columns)

    # -- masks -------------------------------------------------------------
    @property
    def observation_mask(self) -> np.ndarray:
        """``(T,)`` -- True where a parquet row exists for this boxer."""
        return np.asarray([obs.is_observed for obs in self._resolve()], dtype=bool)

    @property
    def detection_mask(self) -> np.ndarray:
        """``(T,)`` -- True where the row has a matched detection + keypoints."""
        return np.asarray([obs.has_detection for obs in self._resolve()], dtype=bool)

    @property
    def keypoints_mask(self) -> np.ndarray:
        """``(T,)`` -- True where at least one keypoint has confidence > 0."""
        kps = self.kps
        if kps.size == 0:
            return np.empty((0,), dtype=bool)
        return (kps[:, :, 2] > 0).any(axis=1)

    @property
    def data(self) -> SegmentData:
        return SegmentData(
            frames=self.frames,
            bbox=self.bbox,
            kps=self.kps,
            meta=self.meta,
            observation_mask=self.observation_mask,
            detection_mask=self.detection_mask,
        )


class SegmentCollection:
    """Several :class:`BoxerSegment` keyed by global id."""

    def __init__(self, segments: dict):
        self._segments = {_as_id(gid): seg for gid, seg in segments.items()}

    @property
    def global_ids(self) -> list:
        return sorted(self._segments.keys())

    def __getitem__(self, global_id) -> BoxerSegment:
        return self._segments[_as_id(global_id)]

    def __iter__(self) -> Iterator[BoxerSegment]:
        for gid in self.global_ids:
            yield self._segments[gid]

    def __len__(self) -> int:
        return len(self._segments)

    def __contains__(self, global_id) -> bool:
        return _as_id(global_id) in self._segments

    def __repr__(self) -> str:
        return f"SegmentCollection(global_ids={self.global_ids})"
