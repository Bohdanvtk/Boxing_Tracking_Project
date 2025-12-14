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
    """
    Center the skeleton around the root joint.

    Parameters
    ----------
    keypoints : np.ndarray, shape (N, 2) or (N, 3)
        Keypoints for every joint. If shape is (N, 3), it is assumed [x, y, conf].
    root_index : int
        Index of the root joint (e.g. BODY_25["MidHip"]).

    Returns
    -------
    centered_keypoints : np.ndarray, shape (N, 2)
        Keypoints with root at (0, 0).
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)

    # Use only x, y
    if keypoints.shape[1] >= 2:
        keypoints_xy = keypoints[:, :2]
    else:
        raise ValueError("keypoints must have at least 2 columns (x, y).")

    root = keypoints_xy[root_index]  # (2,)
    centered = keypoints_xy - root   # (N, 2)
    return centered


def compute_scale_from_shoulders(keypoints, left_shoulder_idx, right_shoulder_idx, min_scale=1e-6):
    """
    Compute a scale factor based on the distance between shoulders.

    Parameters
    ----------
    keypoints : np.ndarray, shape (N, 2)
        Centered keypoints (x, y) for each joint.
    left_shoulder_idx : int
        Index of the left shoulder joint.
    right_shoulder_idx : int
        Index of the right shoulder joint.
    min_scale : float
        Minimum scale value to avoid division by zero.

    Returns
    -------
    scale : float
        Distance between shoulders (at least min_scale).
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)

    ls = keypoints[left_shoulder_idx]
    rs = keypoints[right_shoulder_idx]
    shoulder_vec = rs - ls
    dist = np.linalg.norm(shoulder_vec)

    if dist < min_scale:
        dist = min_scale
    return dist


def scale_skeleton_2d(keypoints, scale):
    """
    Scale the skeleton by dividing all coordinates by 'scale'.

    Parameters
    ----------
    keypoints : np.ndarray, shape (N, 2)
        Centered keypoints.
    scale : float
        Positive scale factor.

    Returns
    -------
    scaled_keypoints : np.ndarray, shape (N, 2)
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    return keypoints / float(scale)


def compute_shoulder_angle(keypoints, left_shoulder_idx, right_shoulder_idx):
    """
    Compute the angle (in radians) between the x-axis and the vector
    from left shoulder to right shoulder.

    Parameters
    ----------
    keypoints : np.ndarray, shape (N, 2)
        Centered & scaled keypoints.
    left_shoulder_idx : int
        Index of the left shoulder joint.
    right_shoulder_idx : int
        Index of the right shoulder joint.

    Returns
    -------
    angle : float
        Angle in radians.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)

    ls = keypoints[left_shoulder_idx]
    rs = keypoints[right_shoulder_idx]
    v = rs - ls  # vector from left shoulder to right shoulder
    angle = np.arctan2(v[1], v[0])  # atan2(y, x)
    return angle


def rotate_skeleton_2d(keypoints, angle):
    """
    Rotate all keypoints by -angle to make the shoulder line horizontal.

    Parameters
    ----------
    keypoints : np.ndarray, shape (N, 2)
        Centered & scaled keypoints.
    angle : float
        Angle in radians that we want to remove.

    Returns
    -------
    rotated_keypoints : np.ndarray, shape (N, 2)
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)

    c = np.cos(-angle)
    s = np.sin(-angle)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)  # 2x2 rotation matrix
    rotated = keypoints @ R.T  # (N, 2) @ (2, 2) -> (N, 2)
    return rotated


def normalize_pose_2d(
        keypoints,
        root_index=BODY_25["MidHip"],
        left_shoulder_idx=BODY_25["LShoulder"],
        right_shoulder_idx=BODY_25["RShoulder"],
        eps=1e-6
):
    """
    Full 2D pose normalization pipeline:
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

    # 1. Centering
    centered = center_skeleton_2d(keypoints, root_index)

    # 2. Scale
    scale = compute_scale_from_shoulders(centered, left_shoulder_idx, right_shoulder_idx, min_scale=eps)
    scaled = scale_skeleton_2d(centered, scale)

    # 3. Shoulder angle
    angle = compute_shoulder_angle(scaled, left_shoulder_idx, right_shoulder_idx)

    # 4. Rotate by -angle
    rotated = rotate_skeleton_2d(scaled, angle)

    return rotated, scale, angle
