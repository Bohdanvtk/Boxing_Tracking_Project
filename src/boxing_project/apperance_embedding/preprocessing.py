import numpy as np
import tensorflow as tf


def pad_to_square_tf(x: tf.Tensor) -> tf.Tensor:
    """
    x: (H,W,3)
    returns: (S,S,3) padded to square with zeros
    """
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    s = tf.maximum(h, w)

    pad_h = s - h
    pad_w = s - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    x = tf.pad(
        x,
        paddings=[[top, bottom], [left, right], [0, 0]],
        mode="CONSTANT",
        constant_values=0.0,
    )
    return x


def preprocess_crops_np(
    imgs: np.ndarray,
    image_size: tuple[int, int] = (128, 128),
    to_rgb: bool = False,
) -> np.ndarray:
    """
    imgs: (N, H, W, 3) uint8 or float32
    Returns: (N, image_size[0], image_size[1], 3) float32 in [0,1]
    """
    if imgs.dtype != np.float32:
        imgs = imgs.astype(np.float32)

    if imgs.max() > 1.5:  # assume uint8 in [0..255]
        imgs = imgs / 255.0

    if to_rgb:
        imgs = imgs[..., ::-1]  # BGR -> RGB

    x = tf.convert_to_tensor(imgs, dtype=tf.float32)


    x = tf.map_fn(pad_to_square_tf, x, fn_output_signature=tf.float32)

    x = tf.image.resize(x, image_size, method="bilinear", antialias=True)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x.numpy()



def preprocess_crops_tf(
    x: tf.Tensor,
    image_size: tuple[int, int] = (128, 128),
    to_rgb: bool = False,
) -> tf.Tensor:
    """
    x: (B,H,W,3) uint8 або float32
    returns: (B, image_size[0], image_size[1], 3) float32 in [0,1]
    """
    x = tf.cast(x, tf.float32)

    # якщо прийшло uint8 [0..255] -> [0..1]
    x = tf.cond(tf.reduce_max(x) > 1.5, lambda: x / 255.0, lambda: x)

    if to_rgb:
        x = x[..., ::-1]


    x = tf.map_fn(pad_to_square_tf, x, fn_output_signature=tf.float32)

    x = tf.image.resize(x, image_size, method="bilinear", antialias=True)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x
