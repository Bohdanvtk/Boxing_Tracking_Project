import numpy as np


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

