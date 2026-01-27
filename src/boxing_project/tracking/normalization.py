import numpy as np

# Indexes for joints in OpenPose BODY_25
BODY_25 = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,       # root
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24
}


def center_skeleton_2d(keypoints, root_index):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    keypoints_xy = keypoints[:, :2]

    # якщо індекс не влазить — не падаємо
    if not (0 <= root_index < keypoints_xy.shape[0]):
        root = np.nanmean(keypoints_xy, axis=0)
    else:
        root = keypoints_xy[root_index]

        # якщо root (0,0) або NaN — fallback
        if not np.isfinite(root).all() or (np.abs(root).sum() < 1e-6):
            root = np.nanmean(keypoints_xy, axis=0)

    centered = keypoints_xy - root
    return centered




def normalize_pose_2d(
        keypoints,
        root_index=BODY_25["MidHip"],
        left_shoulder_idx=BODY_25["LShoulder"],
        right_shoulder_idx=BODY_25["RShoulder"],
        eps=1e-6
):
    """
    Full 2D pose normalization pipeline:
      0) Replacing NaN and pos/neg inf with 0.0
      1) Center around root (MidHip).
      2) Normalize scale based on shoulder distance.
      3) Rotate so that the shoulder line becomes horizontal.

    Parameters
    ----------
    keypoints : np.ndarray, shape (N, 2) or (N, 3)
        Original keypoints for BODY_25 joints.
    root_index : int
        Root joint index (default: MidHip).
    left_shoulder_idx : int
        Left shoulder index.
    right_shoulder_idx : int
        Right shoulder index.
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    norm_keypoints : np.ndarray, shape (N, 2)
        Normalized skeleton (centered, scaled, rotated).
    scale : float
        Scale factor that was used.
    angle : float
        Original shoulder angle before rotation.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)

    # 0. Replacing NaN and pos/neg inf with 0.0
    keypoints[:, :2] = np.nan_to_num(keypoints[:, :2], nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Centering
    centered = center_skeleton_2d(keypoints, root_index)



    return centered

