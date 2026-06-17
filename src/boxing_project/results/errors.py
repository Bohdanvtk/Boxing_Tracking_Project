"""Exceptions raised by the results API."""
from __future__ import annotations

import pandas as pd


class AmbiguousObservationError(Exception):
    """Raised when a single ``(global_track_id, frame_idx)`` maps to many rows.

    This usually means several local tracks were matched to the same global id
    on the same frame. The API refuses to silently pick the first row
    (no ``iloc[0]``).
    """

    def __init__(self, global_track_id, frame_idx: int, rows: pd.DataFrame):
        pairs = list(
            zip(
                rows.get("epoch_id", pd.Series(dtype=object)).tolist(),
                rows.get("local_track_id", pd.Series(dtype=object)).tolist(),
            )
        )
        message = (
            f"Ambiguous observation for global_track_id={global_track_id} "
            f"frame_idx={frame_idx}: {len(rows)} rows match. "
            f"(epoch_id, local_track_id) pairs: {pairs}"
        )
        super().__init__(message)
        self.global_track_id = global_track_id
        self.frame_idx = int(frame_idx)
        self.rows = rows
