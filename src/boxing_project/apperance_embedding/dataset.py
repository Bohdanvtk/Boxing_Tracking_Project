from dataclasses import dataclass
import numpy as np

from .preprocessing import preprocess_crops_np


@dataclass
class CropPairs:
    """
    img_a: (N, H, W, 3) uint8/float
    img_b: (N, H, W, 3) uint8/float
    labels: (N,) 0/1
    """
    img_a: np.ndarray
    img_b: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_npz(cls, path: str) -> "CropPairs":
        data = np.load(path)
        return cls(
            img_a=data["img_a"],
            img_b=data["img_b"],
            labels=data["label"],
        )


def prepare_contrastive_arrays(
    pairs: CropPairs,
    image_size: tuple[int, int] = (128, 128),
    to_rgb: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X1: (N, H, W, 3) float32 [0,1]
      X2: (N, H, W, 3) float32 [0,1]
      y:  (N,) float32
    """
    X1 = preprocess_crops_np(pairs.img_a, image_size=image_size, to_rgb=to_rgb)
    X2 = preprocess_crops_np(pairs.img_b, image_size=image_size, to_rgb=to_rgb)
    y = pairs.labels.astype(np.float32)
    return X1, X2, y
