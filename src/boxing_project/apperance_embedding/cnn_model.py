from dataclasses import dataclass
import tensorflow as tf


@dataclass
class AppearanceCNNConfig:
    image_size: tuple[int, int] = (128, 128)
    embedding_dim: int = 128
    backbone: str = "mobilenetv3small"  # future-proof
    dropout: float = 0.0
    l2_reg: float = 0.0
    train_backbone: bool = True  # can freeze for warmup if you want


def build_appearance_cnn(cfg: AppearanceCNNConfig) -> tf.keras.Model:
    h, w = cfg.image_size
    inputs = tf.keras.Input(shape=(h, w, 3), name="crop")

    reg = tf.keras.regularizers.l2(cfg.l2_reg) if cfg.l2_reg > 0 else None

    bb = cfg.backbone.lower()
    if bb in ("mobilenetv3small", "mnetv3small", "v3small"):
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )
    elif bb in ("mobilenetv3large", "mnetv3large", "v3large"):
        base = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    base.trainable = bool(cfg.train_backbone)

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    if cfg.dropout > 0:
        x = tf.keras.layers.Dropout(cfg.dropout)(x)

    x = tf.keras.layers.Dense(cfg.embedding_dim, kernel_regularizer=reg, name="proj")(x)
    x = tf.keras.layers.UnitNormalization(axis=-1, name="l2norm")(x)

    return tf.keras.Model(inputs, x, name="appearance_encoder")

