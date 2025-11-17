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


def anti_singleton_loss(
    mse_weight: float = 0.1,
    entropy_penalty: float = 1.0,
    low_prob_penalty: float = 2.0,
    low_prob_threshold: float = 0.25,
):
    """
    Custom loss that directly penalizes 1x cards (singletons) in generated decks.
    
    Combines:
    1. MSE loss on normalized histograms (existing approach)
    2. Entropy penalty: penalizes high entropy (encourages concentration on a few cards)
    3. Low-probability penalty: heavily penalizes cards with very low probability (which would be 1x)
    
    Args:
        mse_weight: Weight for MSE loss component
        entropy_penalty: Weight for entropy penalty (higher = more concentration)
        low_prob_penalty: Weight for low-probability penalty (higher = stronger penalty for 1x cards)
        low_prob_threshold: Probability threshold below which cards are considered "low probability" (likely 1x)
    
    Returns:
        Loss function that can be used in model compilation
    """
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # y_true: (batch, vocab_size) - normalized frequency histogram (0-1 scaled by max copies)
        # y_pred: (batch, vocab_size) - predicted frequency histogram (sigmoid outputs)
        
        # 1. MSE loss (existing approach)
        mse = mse_loss(y_true, y_pred)
        
        # 2. Entropy penalty: penalize high entropy (encourages concentration)
        # High entropy = many cards with similar probability = many 1x cards
        # Low entropy = few cards with high probability = 4x staples
        # Entropy = -sum(p * log(p))
        # We want to minimize entropy, so we add it as a penalty
        epsilon = 1e-9
        entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + epsilon), axis=-1)
        entropy_penalty_term = tf.reduce_mean(entropy) * entropy_penalty
        
        # 3. Low-probability penalty: heavily penalize cards with very low probability
        # Cards with prob < threshold are likely to be 1x in generated decks
        # We want to minimize the number of such cards
        low_prob_mask = tf.cast(y_pred < low_prob_threshold, tf.float32)
        # Count how many cards have low probability
        num_low_prob = tf.reduce_sum(low_prob_mask, axis=-1)
        # Penalize: more low-prob cards = higher loss
        # Normalize by vocab size to make it scale-independent
        vocab_size = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        low_prob_penalty_term = tf.reduce_mean(num_low_prob / vocab_size) * low_prob_penalty
        
        total_loss = mse_weight * mse + entropy_penalty_term + low_prob_penalty_term
        
        return total_loss
    
    return _loss