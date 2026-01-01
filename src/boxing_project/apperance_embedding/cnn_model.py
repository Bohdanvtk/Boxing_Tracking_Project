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
    """
    Returns encoder model:
      (H,W,3) -> (embedding_dim,)
    Embeddings are L2-normalized => cosine distance works well.
    """
    h, w = cfg.image_size
    inputs = tf.keras.Input(shape=(h, w, 3), name="crop")

    reg = tf.keras.regularizers.l2(cfg.l2_reg) if cfg.l2_reg > 0 else None

    if cfg.backbone.lower() == "mobilenetv3small":
        base = tf.keras.applications.MobileNetV3Small(
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

    # L2 normalize embedding
    x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm")(x)

    return tf.keras.Model(inputs, x, name="appearance_encoder")
