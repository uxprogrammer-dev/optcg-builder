"""
Custom loss and metric utilities for deck generation training.
"""

from __future__ import annotations

import tensorflow as tf


def masked_sparse_categorical_crossentropy(pad_token_id: int):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
        loss = loss_fn(y_true, y_pred)
        loss *= mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-9)

    return _loss


def masked_accuracy(pad_token_id: int):
    def _metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        matches = tf.cast(tf.equal(y_true, y_pred_ids), tf.float32)
        mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
        matches *= mask
        return tf.reduce_sum(matches) / (tf.reduce_sum(mask) + 1e-9)

    return _metric

