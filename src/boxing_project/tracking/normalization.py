import numpy as np
from typing import Optional, Sequence, Tuple


def center_skeleton_2d(keypoints, root_index):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    xy = keypoints[:, :2]

    # 1) беремо root
    if 0 <= root_index < xy.shape[0]:
        root = xy[root_index]
    else:
        root = np.array([np.nan, np.nan], dtype=np.float32)

    # 2) якщо root NaN / inf / (0,0) → fallback
    if not np.isfinite(root).all() or np.abs(root).sum() < 1e-6:
        return xy.copy()

    # 3) якщо ВСЕ було NaN → просто нічого не ламаємо
    if not np.isfinite(root).all():
        return xy.copy()

    # 4) центрування (NaN залишаються NaN)
    return xy - root


def normalize_pose_2d(
        keypoints,
        root_index,
):
    '''
    Full 2D pose normalization pipeline:
     Center around root (MidHip).

    Parameters
    ----------
    keypoints : np.ndarray, shape (N, 2) or (N, 3)
        Original keypoints for BODY_25 joints.
    root_index : int
        Root joint index (default: MidHip).o.

    Returns
    -------
    norm_keypoints : np.ndarray, shape (N, 2)
        Normalized skeleton (centered, scaled, rotated).
    '''

    keypoints = np.asarray(keypoints, dtype=np.float32)

    # 1. Centering
    centered = center_skeleton_2d(keypoints, root_index)

    return centered


def bones_vector(kpt: np.ndarray, bones: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    kpt = np.asarray(kpt, dtype=np.float32)
    bone_ids = np.asarray(bones, dtype=np.int32).reshape(-1, 2)

    if bone_ids.size == 0:
        return np.zeros((0, 2), dtype=np.float32), bone_ids

    delta = kpt[bone_ids[:, 1], :2] - kpt[bone_ids[:, 0], :2]
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    dirs = delta / (norm + 1e-9)
    return dirs.astype(np.float32), bone_ids


def mirror_invariant(
    kpt: np.ndarray,
    conf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    swapped_kpt = np.asarray(kpt, dtype=np.float32).copy()
    swapped_conf = None if conf is None else np.asarray(conf, dtype=np.float32).copy()

    # BODY_25 label swap for left/right robustness (no geometric mirroring).
    swap_pairs = ((2, 5), (3, 6), (9, 12), (10, 13))
    for left, right in swap_pairs:
        swapped_kpt[[left, right]] = swapped_kpt[[right, left]]
        if swapped_conf is not None:
            swapped_conf[[left, right]] = swapped_conf[[right, left]]

    return swapped_kpt, swapped_conf
