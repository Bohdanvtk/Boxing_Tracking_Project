import numpy as np
import tensorflow as tf


def preprocess_crops_np(
    imgs: np.ndarray,
    image_size: tuple[int, int] = (128, 128),
    to_rgb: bool = False,
) -> np.ndarray:
    """
    imgs: (N, H, W, 3) uint8 or float32
    Returns: (N, image_size[0], image_size[1], 3) float32 in [0,1]

    to_rgb:
      - if your crops are BGR (from OpenCV), set to_rgb=True to swap channels.
      - if your crops are already RGB, keep False.
    """
    if imgs.dtype != np.float32:
        imgs = imgs.astype(np.float32)

    if imgs.max() > 1.5:  # assume uint8 in [0..255]
        imgs = imgs / 255.0

    if to_rgb:
        imgs = imgs[..., ::-1]  # BGR -> RGB

    # resize with TF (fast, good quality)
    x = tf.convert_to_tensor(imgs, dtype=tf.float32)
    x = tf.image.resize(x, image_size, method="bilinear", antialias=True)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x.numpy()
