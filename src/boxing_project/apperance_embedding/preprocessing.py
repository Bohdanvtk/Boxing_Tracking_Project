from __future__ import annotations

import cv2
import numpy as np


def preprocess_crops_np(
    imgs: np.ndarray,
    image_size: tuple[int, int] = (256, 128),
    to_rgb: bool = True,
) -> np.ndarray:
    """
    OSNet/ReID preprocessing.

    Args:
        imgs: (N, H, W, 3) uint8/float32 in BGR by default.
        image_size: (H, W), default ReID format (256, 128).
        to_rgb: convert BGR -> RGB before normalization.

    Returns:
        np.ndarray (N, 3, H, W) float32 normalized by ImageNet mean/std.
    """
    imgs = np.asarray(imgs)
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(f"Expected imgs with shape (N,H,W,3), got {imgs.shape}")

    out_h, out_w = image_size
    n = imgs.shape[0]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    processed = np.empty((n, 3, out_h, out_w), dtype=np.float32)

    for i in range(n):
        crop = imgs[i]
        if crop.dtype != np.float32:
            crop = crop.astype(np.float32)

        if crop.max() > 1.5:  # uint8-like [0..255]
            crop = crop / 255.0

        crop = np.clip(crop, 0.0, 1.0)

        if to_rgb:
            crop = crop[..., ::-1]  # BGR -> RGB

        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        crop = (crop - mean) / std
        processed[i] = np.transpose(crop, (2, 0, 1))

    return processed


def preprocess_crops_tf(
    x,
    image_size: tuple[int, int] = (256, 128),
    to_rgb: bool = True,
) -> np.ndarray:
    """
    Backward-compatible alias without TensorFlow dependency.
    Accepts numpy arrays or tensor-like objects with `.numpy()`.
    """
    if hasattr(x, "numpy"):
        x = x.numpy()
    return preprocess_crops_np(np.asarray(x), image_size=image_size, to_rgb=to_rgb)
