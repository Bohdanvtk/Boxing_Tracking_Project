import tensorflow as tf


def contrastive_loss(margin: float = 1.0):
    """
    Contrastive loss:
      y=1 -> positive -> minimize distance
      y=0 -> negative -> push distance >= margin

    y_true: (B,) float 0/1
    y_pred: (B,1) distance
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        d = tf.squeeze(y_pred, axis=-1)  # (B,)

        pos = y_true * tf.square(d)
        neg = (1.0 - y_true) * tf.square(tf.maximum(margin - d, 0.0))
        return tf.reduce_mean(pos + neg)

    return loss
