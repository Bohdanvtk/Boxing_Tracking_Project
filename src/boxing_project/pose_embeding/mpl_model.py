# src/boxing_project/pose_embeding/mlp_model.py

from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


@dataclass
class PoseMLPConfig:
    num_keypoints: int
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    dropout: float = 0.0
    l2_reg: float = 0.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


def build_pose_mlp(config: PoseMLPConfig) -> Model:
    """
    Build a simple MLP that maps flattened normalized pose (2*K) -> embedding_dim.
    """
    input_dim = config.num_keypoints * 2

    inputs = layers.Input(shape=(input_dim,), name="pose_input")

    x = inputs
    for i, h in enumerate(config.hidden_dims):
        x = layers.Dense(
            h,
            activation="relu",
            kernel_regularizer=regularizers.l2(config.l2_reg),
            name=f"dense_{i}"
        )(x)
        if config.dropout > 0.0:
            x = layers.Dropout(config.dropout, name=f"dropout_{i}")(x)

    # Final embedding
    x = layers.Dense(
        config.embedding_dim,
        activation=None,
        kernel_regularizer=regularizers.l2(config.l2_reg),
        name="embedding"
    )(x)

    # L2-normalize to use cosine similarity nicely
    outputs = tf.nn.l2_normalize(x, axis=-1, name="l2_normalized_embedding")

    model = Model(inputs=inputs, outputs=outputs, name="pose_mlp")
    return model
