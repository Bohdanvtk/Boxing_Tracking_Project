import numpy as np
import tensorflow as tf


def frame_to_gray_tensor(
    frame_bgr: np.ndarray,
    target_size: tuple[int, int] = (160, 90),
) -> tf.Tensor:
    """
    frame_bgr: (H, W, 3) uint8 або float
    return: (1, target_h, target_w, 1) float32 у [0,1] (grayscale)

    Ми робимо:
      - uint8 -> float32
      - /255
      - resize (швидко + стабільно)
      - BGR -> RGB (бо tf.image.rgb_to_grayscale очікує RGB)
      - grayscale
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3:
        raise ValueError(f"Expected frame shape (H,W,3), got {frame_bgr.shape}")

    x = frame_bgr.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0

    # BGR -> RGB
    x = x[..., ::-1]

    x = tf.convert_to_tensor(x, dtype=tf.float32)  # (H,W,3)
    x = tf.expand_dims(x, axis=0)                  # (1,H,W,3)

    x = tf.image.resize(x, target_size, method="bilinear", antialias=True)
    x = tf.clip_by_value(x, 0.0, 1.0)

    x = tf.image.rgb_to_grayscale(x)               # (1,h,w,1)
    return x
