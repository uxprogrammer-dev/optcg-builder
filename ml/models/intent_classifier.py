"""
Transformer-based encoder-only model for intent classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import tensorflow as tf
from tensorflow.keras import Model, layers

from .transformer import PositionalEncodingLayer, TransformerConfig, TransformerEncoderLayer

# Try to import register_keras_serializable from various locations
try:
    from tensorflow.keras.saving import register_keras_serializable
except (ImportError, AttributeError):
    try:
        from keras.saving import register_keras_serializable
    except (ImportError, AttributeError):
        try:
            from tensorflow.keras.utils import register_keras_serializable
        except (ImportError, AttributeError):
            # If all else fails, create a no-op decorator
            def register_keras_serializable(package=None, name=None):
                def decorator(cls):
                    return cls
                return decorator


@dataclass
class IntentClassifierConfig:
    """Configuration for intent classifier model."""

    embedding_dim: int = 256
    num_encoder_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 512
    dropout_rate: float = 0.1


def build_intent_classifier(
    prompt_vocab_size: int,
    prompt_max_length: int,
    num_colors: int = 6,
    num_keywords: int = 1000,
    num_leaders: int = 245,
    num_strategies: int = 5,
    config: IntentClassifierConfig = None,
) -> Model:
    """
    Build an encoder-only transformer model for intent classification.
    
    Args:
        prompt_vocab_size: Size of prompt vocabulary
        prompt_max_length: Maximum prompt sequence length
        num_colors: Number of color classes (default: 6)
        num_keywords: Number of keyword classes
        num_leaders: Number of leader classes
        num_strategies: Number of strategy tag classes
        config: Model configuration (uses defaults if None)
    
    Returns:
        Keras Model with multi-task classification heads
    """
    if config is None:
        config = IntentClassifierConfig()

    transformer_config = TransformerConfig(
        embedding_dim=config.embedding_dim,
        ff_dim=config.ff_dim,
        num_heads=config.num_heads,
        num_layers=config.num_encoder_layers,
        dropout_rate=config.dropout_rate,
    )

    # Input: prompt tokens
    prompt_inputs = layers.Input(shape=(None,), dtype=tf.int32, name="prompt_tokens")

    # Embedding layer
    prompt_embedding = layers.Embedding(
        input_dim=prompt_vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="prompt_embedding",
    )(prompt_inputs)

    # Add positional encoding
    prompt_embedding = PositionalEncodingLayer(
        max_length=prompt_max_length,
        embedding_dim=config.embedding_dim,
        name="add_pos_encoding",
    )(prompt_embedding)

    # Apply dropout
    x = layers.Dropout(config.dropout_rate, name="embedding_dropout")(prompt_embedding)

    # Create padding mask
    padding_mask = layers.Lambda(
        lambda tokens: tf.cast(tf.not_equal(tokens, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :],
        name="create_padding_mask",
    )(prompt_inputs)

    # Encoder layers
    for i in range(config.num_encoder_layers):
        x = TransformerEncoderLayer(transformer_config, name=f"encoder_layer_{i}")(
            x, training=None, mask=padding_mask
        )

    # Global average pooling to get sequence-level representation
    pooled_output = layers.GlobalAveragePooling1D(name="global_pool")(x)

    # Multi-task classification heads
    # Colors: binary classification for each color
    color_output = layers.Dense(
        num_colors,
        activation="sigmoid",
        name="colors",
    )(pooled_output)

    # Keywords: binary classification for each keyword
    keyword_output = layers.Dense(
        num_keywords,
        activation="sigmoid",
        name="keywords",
    )(pooled_output)

    # Leaders: binary classification for each leader (with scores for ranking)
    leader_output = layers.Dense(
        num_leaders,
        activation="sigmoid",
        name="leaders",
    )(pooled_output)

    # Strategies: binary classification for each strategy tag
    strategy_output = layers.Dense(
        num_strategies,
        activation="sigmoid",
        name="strategies",
    )(pooled_output)

    # Build model
    model = Model(
        inputs=[prompt_inputs],
        outputs={
            "colors": color_output,
            "keywords": keyword_output,
            "leaders": leader_output,
            "strategies": strategy_output,
        },
        name="intent_classifier",
    )

    return model

