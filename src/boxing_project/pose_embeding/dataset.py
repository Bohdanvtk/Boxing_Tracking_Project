# src/boxing_project/pose_embeding/dataset.py

from dataclasses import dataclass
import numpy as np
from .normalization import normalize_pose_2d


@dataclass
class PosePairs:
    pose_a: np.ndarray  # (N, K, 3) or (N, K, 2)
    pose_b: np.ndarray  # (N, K, 3) or (N, K, 2)
    labels: np.ndarray  # (N,)

    @classmethod
    def from_npz(cls, path: str) -> "PosePairs":
        data = np.load(path)
        return cls(
            pose_a=data["pose_a"],
            pose_b=data["pose_b"],
            labels=data["label"],
        )


def normalize_and_flatten_all(pose: np.ndarray) -> np.ndarray:
    """
    pose: (N, K, 2 or 3)
    return: (N, 2*K) float32
    """
    N, K, D = pose.shape
    out = np.zeros((N, K * 2), dtype=np.float32)

    for i in range(N):
        kp = pose[i]
        norm_kp, _, _ = normalize_pose_2d(kp)   # expected (K,2)
        out[i] = norm_kp.reshape(-1)

    return out


def prepare_contrastive_arrays(pairs: PosePairs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X1: (N, 2K)
      X2: (N, 2K)
      y:  (N,) float32
    """
    X1 = normalize_and_flatten_all(pairs.pose_a)
    X2 = normalize_and_flatten_all(pairs.pose_b)
    y = pairs.labels.astype(np.float32)
    return X1, X2, y
