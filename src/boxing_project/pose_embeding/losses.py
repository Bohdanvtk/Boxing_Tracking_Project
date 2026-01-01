# src/boxing_project/pose_embeding/losses.py

import tensorflow as tf


def contrastive_loss(margin: float = 1.0):
    """
    Contrastive loss for metric learning.

    y_true: (B,)    -> 1.0 positive, 0.0 negative
    y_pred: (B, 1)  -> distance between embeddings
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        d = tf.squeeze(y_pred, axis=-1)  # (B,)

        pos_loss = y_true * tf.square(d)
        neg_loss = (1.0 - y_true) * tf.square(tf.maximum(margin - d, 0.0))

        return tf.reduce_mean(pos_loss + neg_loss)

    return loss
