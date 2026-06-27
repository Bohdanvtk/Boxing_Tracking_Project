"""Internal helpers shared across the results package.

Nothing here is part of the public API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import N_KEYPOINTS


def _is_missing(value) -> bool:
    """Return True for null scalars, treating lists/arrays as present.

    Handles ``None``, ``float('nan')``, ``np.nan``, NumPy scalar NaN, ``pd.NA``
    and other pandas nullable scalars. List and array keypoint payloads always
    count as present, so :func:`pandas.isna` is never called on them (it would
    return an elementwise boolean array, not a single truth value).
    """
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_id(value):
    """Normalise a global/track id to a plain ``int`` when possible."""
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _as_list(value) -> list:
    """Normalise a scalar-or-iterable argument into a plain list.

    Strings are treated as scalars (wrapped, not iterated).
    """
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    return [value]


def empty_keypoints() -> np.ndarray:
    """Return a ``(25, 3)`` array of NaN coordinates with zero confidence."""
    out = np.full((N_KEYPOINTS, 3), np.nan, dtype=float)
    out[:, 2] = 0.0
    return out


def row_keypoints(row: pd.Series) -> np.ndarray:
    """Combine ``keypoints`` and ``kp_conf`` into a ``(25, 3)`` array.

    Columns are ``x``, ``y``, ``confidence``. When the detection is missing the
    coordinates are NaN and the confidence is ``0.0`` (no interpolation, no
    forward-fill, no copying from a neighbouring frame).

    Raises:
        ValueError: When the stored keypoints/confidence do not have the
            expected BODY_25 shape.
    """
    out = empty_keypoints()

    kp = row["keypoints"] if "keypoints" in row else None
    if not _is_missing(kp):
        kp_list = list(kp)
        if len(kp_list) != N_KEYPOINTS:
            raise ValueError(
                f"Expected {N_KEYPOINTS} BODY_25 keypoints, received {len(kp_list)}"
            )
        xy = np.asarray([np.asarray(point, dtype=float) for point in kp_list], dtype=float)
        if xy.shape != (N_KEYPOINTS, 2):
            raise ValueError(
                f"Expected keypoints of shape ({N_KEYPOINTS}, 2), received {xy.shape}"
            )
        out[:, :2] = xy

    conf = row["kp_conf"] if "kp_conf" in row else None
    if not _is_missing(conf):
        conf_arr = np.asarray(list(conf), dtype=float)
        if conf_arr.shape != (N_KEYPOINTS,):
            raise ValueError(
                f"Expected kp_conf with {N_KEYPOINTS} values, received {conf_arr.size}"
            )
        out[:, 2] = conf_arr

    return out
