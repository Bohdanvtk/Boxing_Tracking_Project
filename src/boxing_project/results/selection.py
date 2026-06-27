"""Filtering surface over the observations table.

:class:`TrackSelection` is a filtered, read-only view; :class:`_Query` is the
shared base that :class:`TrackSelection` and ``BoxingResults`` both build on.
Every filter returns a new view and never mutates the source.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .errors import AmbiguousObservationError
from .observation import FrameObservation
from .segment import BoxerSegment, SegmentCollection
from .utils import _as_id, _as_list


class _Query:
    """Shared read-only filtering surface for results and selections.

    Every filter returns a :class:`TrackSelection` over a filtered DataFrame.
    Filtering relies on boolean indexing, which yields new DataFrames; the
    original data is never modified in place.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    # -- pandas access -----------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        """Raw backing DataFrame.

        Provided as an escape hatch for pandas operations not covered by the
        API. Mutating the returned object may affect this view's internal
        state -- use :meth:`to_pandas` when you need an independent copy.
        """
        return self._df

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        """Return the backing rows; an independent copy by default."""
        return self._df.copy() if copy else self._df

    def __len__(self) -> int:
        return len(self._df)

    def _new(self, df: pd.DataFrame) -> "TrackSelection":
        return TrackSelection(df)

    def _focus_global_ids(self) -> list:
        if "global_track_id" not in self._df.columns:
            return []
        ids = self._df["global_track_id"].dropna().unique()
        return sorted(_as_id(value) for value in ids)

    # -- identity filters --------------------------------------------------
    def global_id(self, global_id) -> "TrackSelection":
        return self._new(self._df[self._df["global_track_id"] == global_id])

    def global_ids(self, ids: Iterable) -> "TrackSelection":
        return self._new(self._df[self._df["global_track_id"].isin(_as_list(ids))])

    def epoch(self, epoch_id) -> "TrackSelection":
        return self._new(self._df[self._df["epoch_id"] == epoch_id])

    def epochs(self, epoch_ids: Iterable) -> "TrackSelection":
        return self._new(self._df[self._df["epoch_id"].isin(_as_list(epoch_ids))])

    def local_track(self, local_track_id, epoch_id) -> "TrackSelection":
        """Filter by a ``(local_track_id, epoch_id)`` pair.

        ``local_track_id`` is only unique within an epoch, so the epoch is
        required.
        """
        mask = (self._df["local_track_id"] == local_track_id) & (
            self._df["epoch_id"] == epoch_id
        )
        return self._new(self._df[mask])

    # -- frame filters -----------------------------------------------------
    def at_frame(self, frame_idx: int) -> "TrackSelection":
        return self._new(self._df[self._df["frame_idx"] == frame_idx])

    def frame(self, frame_idx: int) -> FrameObservation:
        """Resolve a single :class:`FrameObservation` at ``frame_idx``.

        Requires the selection to resolve to exactly one row at that frame.
        Multiple rows for one global id raise :class:`AmbiguousObservationError`;
        rows for several different global ids ask the caller to narrow first.
        """
        sub = self._df[self._df["frame_idx"] == frame_idx]
        if len(sub) == 0:
            raise LookupError(
                f"No observation at frame_idx={frame_idx} for this selection."
            )
        if len(sub) == 1:
            row = sub.iloc[0]
            return FrameObservation(frame_idx, _as_id(row.get("global_track_id")), row)

        distinct = set(sub["global_track_id"].tolist())
        if len(distinct) <= 1:
            gid = _as_id(next(iter(distinct))) if distinct else None
            raise AmbiguousObservationError(gid, frame_idx, sub)
        raise ValueError(
            f"Multiple global ids at frame_idx={frame_idx}: "
            f"{sorted(_as_id(v) for v in distinct)}. Narrow with global_id(...) first."
        )

    def frames(self, start: int, end: int):
        """Observations within ``[start, end]`` (both bounds inclusive).

        Return type depends on how many global ids are in scope:

        * exactly one focused global id -> :class:`BoxerSegment`;
        * zero or several -> :class:`TrackSelection`.

        Only rows that really exist in the range are returned; the length need
        not equal the number of frame numbers in ``[start, end]`` (no padding).
        """
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        mask = (self._df["frame_idx"] >= start) & (self._df["frame_idx"] <= end)
        sub = self._df[mask]
        focus = self._focus_global_ids()
        if len(focus) == 1:
            gid = focus[0]
            return BoxerSegment(gid, sub[sub["global_track_id"] == gid], frame_axis=None)
        return self._new(sub)

    def window(self, start_frame: int, length: int):
        """Exactly ``length`` consecutive time positions from ``start_frame``.

        Missing frames are padded (``observation_mask=False``), so the result is
        suitable as fixed-length model input. Return type depends on scope:

        * exactly one focused global id -> :class:`BoxerSegment`;
        * several focused global ids -> :class:`SegmentCollection` (one padded
          segment per boxer over the same axis).

        Raises ``ValueError`` when no global id is in scope (padding needs a
        target global id) or when ``length <= 0``.
        """
        if length <= 0:
            raise ValueError("length must be greater than zero")
        axis = list(range(int(start_frame), int(start_frame) + int(length)))
        focus = self._focus_global_ids()
        if len(focus) == 1:
            gid = focus[0]
            rows = self._df[self._df["global_track_id"] == gid]
            return BoxerSegment(gid, rows, frame_axis=axis)
        if len(focus) == 0:
            raise ValueError(
                "window() needs a single global id in scope; call global_id(...) first."
            )
        segments = {
            gid: BoxerSegment(
                gid, self._df[self._df["global_track_id"] == gid], frame_axis=axis
            )
            for gid in focus
        }
        return SegmentCollection(segments)

    # -- boolean filters ---------------------------------------------------
    def matched_only(self) -> "TrackSelection":
        return self._new(self._df[self._df["is_matched"] == True])  # noqa: E712

    def confirmed_only(self) -> "TrackSelection":
        return self._new(self._df[self._df["confirmed"] == True])  # noqa: E712

    def with_keypoints(self) -> "TrackSelection":
        from .utils import _is_missing

        mask = self._df["keypoints"].map(lambda v: not _is_missing(v))
        return self._new(self._df[mask])

    def unassigned(self) -> "TrackSelection":
        return self._new(self._df[self._df["global_track_id"].isna()])

    # -- general select ----------------------------------------------------
    def select(
        self,
        global_ids=None,
        epoch_ids=None,
        local_track_ids=None,
        frame_range=None,
        matched=None,
        confirmed=None,
    ) -> "TrackSelection":
        """Apply several filters at once. Scalars and lists are both accepted."""
        df = self._df
        if global_ids is not None:
            df = df[df["global_track_id"].isin(_as_list(global_ids))]
        if epoch_ids is not None:
            df = df[df["epoch_id"].isin(_as_list(epoch_ids))]
        if local_track_ids is not None:
            df = df[df["local_track_id"].isin(_as_list(local_track_ids))]
        if frame_range is not None:
            start, end = frame_range
            if end < start:
                raise ValueError(f"frame_range end ({end}) must be >= start ({start})")
            df = df[(df["frame_idx"] >= start) & (df["frame_idx"] <= end)]
        if matched is not None:
            df = df[df["is_matched"] == bool(matched)]
        if confirmed is not None:
            df = df[df["confirmed"] == bool(confirmed)]
        return self._new(df)

    # -- grouping ----------------------------------------------------------
    def segments(self) -> SegmentCollection:
        """Split the selection by ``global_track_id`` (null ids are skipped)."""
        segments = {}
        if "global_track_id" in self._df.columns:
            for gid, group in self._df.groupby("global_track_id"):
                segments[_as_id(gid)] = BoxerSegment(gid, group, frame_axis=None)
        return SegmentCollection(segments)

    def local_tracks(self) -> pd.DataFrame:
        """Summary of the ``(epoch_id, local_track_id)`` fragments per global id.

        A single global id may span several local fragments across epochs; they
        are grouped by the full ``(global_track_id, epoch_id, local_track_id)``
        key because ``local_track_id`` is not globally unique.

        Columns: ``global_track_id``, ``epoch_id``, ``local_track_id``,
        ``start_frame``, ``end_frame``, ``observations``, ``matched_frames``.
        """
        rows = []
        keys = ["global_track_id", "epoch_id", "local_track_id"]
        for (gid, epoch_id, local_id), group in self._df.groupby(keys):
            matched = group["is_matched"]
            rows.append(
                {
                    "global_track_id": _as_id(gid),
                    "epoch_id": _as_id(epoch_id),
                    "local_track_id": _as_id(local_id),
                    "start_frame": int(group["frame_idx"].min()),
                    "end_frame": int(group["frame_idx"].max()),
                    "observations": int(len(group)),
                    "matched_frames": int((matched == True).sum()),  # noqa: E712
                }
            )
        columns = [
            "global_track_id",
            "epoch_id",
            "local_track_id",
            "start_frame",
            "end_frame",
            "observations",
            "matched_frames",
        ]
        return pd.DataFrame(rows, columns=columns)

    # -- sampling ----------------------------------------------------------
    def sample_per_global(
        self,
        n: int,
        random_state: int | None = None,
        matched_only: bool = False,
    ) -> "TrackSelection":
        """Sample up to ``n`` observations for each non-null global id."""
        if n <= 0:
            raise ValueError("n must be greater than zero")
        df = self._df
        if matched_only:
            df = df[df["is_matched"] == True]  # noqa: E712
        df = df[df["global_track_id"].notna()]
        parts = [
            group.sample(n=min(n, len(group)), random_state=random_state)
            for _, group in df.groupby("global_track_id")
        ]
        sampled = pd.concat(parts).reset_index(drop=True) if parts else df.iloc[0:0]
        return self._new(sampled)


class TrackSelection(_Query):
    """A filtered, read-only view; may contain several global ids."""

    def __repr__(self) -> str:
        return (
            f"TrackSelection(rows={len(self._df)}, "
            f"global_ids={self._focus_global_ids()})"
        )
