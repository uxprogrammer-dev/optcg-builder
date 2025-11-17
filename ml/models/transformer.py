"""
Transformer-based encoder-decoder for prompt→deck generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers

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


TYPE_BUCKETS = ("CHARACTER", "EVENT", "STAGE", "OTHER")
COST_BUCKETS = 11


@register_keras_serializable(package="ml.models.transformer", name="gather_card_features")
def gather_card_features(inputs):
    """
    Registered top-level implementation to support deserializing Lambda layer
    named 'gather_card_features' in saved models.
    """
    projection, token_ids = inputs  # projection -> (vocab_size, embed_dim), token_ids -> (batch, seq_len)
    token_ids = tf.cast(token_ids, tf.int32)
    flat_token_ids = tf.reshape(token_ids, [-1])
    gathered = tf.gather(projection, flat_token_ids, axis=0)
    seq_shape = tf.shape(token_ids)
    embed_dim = tf.shape(projection)[-1]
    gathered = tf.reshape(gathered, tf.concat([seq_shape, [embed_dim]], axis=0))
    return gathered


@dataclass
class TransformerConfig:
    embedding_dim: int = 256
    ff_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout_rate: float = 0.1
    coverage_loss_weight: float = 0.01


def positional_encoding(max_length: int, depth: int) -> tf.Tensor:
    angle_rads = _get_angles(
        tf.range(max_length, dtype=tf.float32)[:, tf.newaxis],
        tf.range(depth, dtype=tf.float32)[tf.newaxis, :],
        depth,
    )
    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding[tf.newaxis, ...]


def _get_angles(pos: tf.Tensor, i: tf.Tensor, depth: int) -> tf.Tensor:
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(depth, tf.float32))
    return pos * angle_rates


@register_keras_serializable(package="ml.models.transformer")
class PositionalEncodingLayer(layers.Layer):
    """Custom layer that adds positional encoding to embeddings dynamically."""

    def __init__(self, max_length: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        # Create positional encoding as a constant
        self.pos_encoding = positional_encoding(max_length, embedding_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        pos_enc_sliced = self.pos_encoding[:, :seq_len, :]
        return inputs + pos_enc_sliced

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_length": self.max_length,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="ml.models.transformer")
class ExtractFeatureLayer(layers.Layer):
    """Extract the first element along axis 0 to remove batch dimension."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.gather(inputs, 0, axis=0)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[1:]


@register_keras_serializable(package="ml.models.transformer")
class PredictedSequenceFreqHistLayer(layers.Layer):
    """
    Phase 1: Compute frequency histogram from argmax predictions of main output.
    
    This layer takes the main output logits, computes argmax predictions,
    and returns a normalized frequency histogram for sequence-level loss.
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        start_token_id: int,
        end_token_id: int,
        max_copies: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.max_copies = max_copies
    
    def call(self, main_logits: tf.Tensor) -> tf.Tensor:
        """
        Compute frequency histogram from argmax predictions.
        
        Args:
            main_logits: (batch, seq_len, vocab_size) - logits from main output
        
        Returns:
            (batch, vocab_size) - normalized frequency histogram [0, 1]
        """
        # Get predicted tokens (argmax)
        predicted_tokens = tf.argmax(main_logits, axis=-1, output_type=tf.int32)  # (batch, seq_len)
        
        # Convert to one-hot
        predicted_one_hot = tf.one_hot(
            predicted_tokens,
            depth=self.vocab_size,
            dtype=tf.float32
        )  # (batch, seq_len, vocab_size)
        
        # Sum across sequence dimension to get frequency histogram
        predicted_freq_hist = tf.reduce_sum(predicted_one_hot, axis=1)  # (batch, vocab_size)
        
        # Mask out special tokens (PAD, BOS, EOS)
        special_ids = tf.constant([self.pad_token_id, self.start_token_id, self.end_token_id], dtype=tf.int32)
        special_mask = tf.ones((self.vocab_size,), dtype=tf.float32)
        # Zero out special tokens
        special_mask = tf.tensor_scatter_nd_update(
            special_mask,
            tf.expand_dims(special_ids, 1),
            tf.zeros((tf.shape(special_ids)[0],), dtype=tf.float32)
        )
        special_mask = tf.expand_dims(special_mask, 0)  # (1, vocab_size)
        predicted_freq_hist = predicted_freq_hist * special_mask
        
        # Normalize by max copies
        max_copies_tensor = tf.constant(float(self.max_copies), dtype=tf.float32)
        predicted_freq_hist = tf.minimum(predicted_freq_hist, max_copies_tensor)
        predicted_freq_hist = tf.math.divide_no_nan(
            predicted_freq_hist,
            max_copies_tensor
        )
        
        return predicted_freq_hist
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "start_token_id": self.start_token_id,
            "end_token_id": self.end_token_id,
            "max_copies": self.max_copies,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="ml.models.transformer")
class PaddingMaskLayer(layers.Layer):
    """Creates padding mask for token sequences: [batch, seq] -> [batch, 1, 1, seq]."""

    def __init__(self, pad_token_id: int, **kwargs):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id

    def call(self, tokens: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.equal(tokens, self.pad_token_id), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, seq_len) possibly with None values
        batch = input_shape[0] if isinstance(input_shape, (tuple, list)) else None
        seq_len = input_shape[1] if isinstance(input_shape, (tuple, list)) else None
        return (batch, 1, 1, seq_len)

    def get_config(self):
        config = super().get_config()
        config.update({"pad_token_id": self.pad_token_id})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="ml.models.transformer")
class CombinedDecoderMaskLayer(layers.Layer):
    """Combines padding mask with look-ahead mask: [b,1,1,S] -> [b,1,S,S]."""

    def call(self, padding_mask: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(padding_mask)[-1]
        look_ahead = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        look_ahead = 1.0 - look_ahead
        look_ahead = tf.cast(look_ahead, tf.float32)
        look_ahead = tf.reshape(look_ahead, (1, 1, seq_len, seq_len))
        return tf.maximum(padding_mask, look_ahead)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, 1, 1, seq_len) possibly with None values
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 4:
            return (None, 1, None, None)
        batch = input_shape[0]
        seq_len = input_shape[3]
        return (batch, 1, seq_len, seq_len)


@register_keras_serializable(package="ml.models.transformer")
class TransformerEncoderLayer(layers.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.ff_dim = config.ff_dim
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.mha = layers.MultiHeadAttention(num_heads=config.num_heads, key_dim=config.embedding_dim)
        self.dropout1 = layers.Dropout(config.dropout_rate)
        self.dropout2 = layers.Dropout(config.dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(config.ff_dim, activation="relu"),
                layers.Dense(config.embedding_dim),
            ]
        )

    def call(self, x: tf.Tensor, training: bool | None = None, mask: tf.Tensor | None = None) -> tf.Tensor:
        attn_output = self.mha(x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "ff_dim": self.ff_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        transformer_config = TransformerConfig(
            embedding_dim=config.pop("embedding_dim", 256),
            ff_dim=config.pop("ff_dim", 512),
            num_heads=config.pop("num_heads", 8),
            dropout_rate=config.pop("dropout_rate", 0.1),
        )
        return cls(transformer_config, **config)


@register_keras_serializable(package="ml.models.transformer")
class TransformerDecoderLayer(layers.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.ff_dim = config.ff_dim
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.coverage_loss_weight = config.coverage_loss_weight
        self.self_attention = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.embedding_dim,
            dropout=config.dropout_rate,
        )
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.embedding_dim,
            dropout=config.dropout_rate,
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(config.ff_dim, activation="relu"),
                layers.Dense(config.embedding_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(config.dropout_rate)
        self.dropout2 = layers.Dropout(config.dropout_rate)
        self.dropout3 = layers.Dropout(config.dropout_rate)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        training: bool | None = None,
        look_ahead_mask: tf.Tensor | None = None,
        padding_mask: tf.Tensor | None = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        attn1, attn_weights_block1 = self.self_attention(
            query=x,
            value=x,
            key=x,
            attention_mask=look_ahead_mask,
            return_attention_scores=True,
        )
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.cross_attention(
            query=out1,
            value=enc_output,
            key=enc_output,
            attention_mask=padding_mask,
            return_attention_scores=True,
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        coverage = tf.reduce_mean(tf.square(tf.reduce_sum(attn_weights_block2, axis=-1) - 1.0))
        self.add_loss(self.coverage_loss_weight * coverage)

        return out3, attn_weights_block1, attn_weights_block2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "ff_dim": self.ff_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "coverage_loss_weight": self.coverage_loss_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        transformer_config = TransformerConfig(
            embedding_dim=config.pop("embedding_dim", 256),
            ff_dim=config.pop("ff_dim", 512),
            num_heads=config.pop("num_heads", 8),
            dropout_rate=config.pop("dropout_rate", 0.1),
            coverage_loss_weight=config.pop("coverage_loss_weight", 0.01),
        )
        return cls(transformer_config, **config)


def build_deck_transformer(
    prompt_vocab_size: int,
    deck_vocab_size: int,
    prompt_sequence_length: int,
    deck_sequence_length: int,
    pad_token_id: int,
    config: TransformerConfig = TransformerConfig(),
    use_card_features: bool = True,
    card_feature_dim: int = 128,
    start_token_id: int = 1,  # Phase 1: For sequence-level loss
    end_token_id: int = 2,  # Phase 1: For sequence-level loss
) -> Model:
    encoder_inputs = layers.Input(shape=(prompt_sequence_length,), dtype="int32", name="prompt_tokens")
    decoder_inputs = layers.Input(shape=(deck_sequence_length,), dtype="int32", name="decoder_input")
    
    # Optional card feature inputs
    card_cost_features = None
    card_power_features = None
    card_color_features = None
    card_type_features = None
    card_ability_features = None
    card_has_ability_features = None
    
    if use_card_features:
        # Card features are provided as constant tensors of shape (vocab_size,)
        # We'll create Input layers that accept these as constants
        # Note: In practice, these will be passed as model constants during training
        # For now, we create placeholder inputs that can be replaced with constants
        card_cost_features = layers.Input(shape=(deck_vocab_size,), dtype="int32", name="card_cost_features")
        card_power_features = layers.Input(shape=(deck_vocab_size,), dtype="int32", name="card_power_features")
        card_color_features = layers.Input(shape=(deck_vocab_size,), dtype="int32", name="card_color_features")
        card_type_features = layers.Input(shape=(deck_vocab_size,), dtype="int32", name="card_type_features")
        card_ability_features = layers.Input(shape=(deck_vocab_size, 64), dtype="int32", name="card_ability_features")
        card_has_ability_features = layers.Input(shape=(deck_vocab_size,), dtype="int32", name="card_has_ability_features")

    # Padding masks using custom serializable layer (no Lambda)
    encoder_padding_mask = PaddingMaskLayer(pad_token_id, name="encoder_padding_mask")(encoder_inputs)
    decoder_padding_mask = PaddingMaskLayer(pad_token_id, name="decoder_padding_mask")(decoder_inputs)

    # Look-ahead + padding combined mask using custom serializable layer
    combined_decoder_mask = CombinedDecoderMaskLayer(name="decoder_look_ahead_mask")(decoder_padding_mask)

    # Create positional encodings with maximum lengths (will be sliced dynamically)
    positional_enc_prompt = positional_encoding(prompt_sequence_length, config.embedding_dim)
    positional_enc_deck = positional_encoding(deck_sequence_length, config.embedding_dim)

    encoder_embedding = layers.Embedding(
        input_dim=prompt_vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=False,
        name="prompt_embedding",
    )(encoder_inputs)

    # Add positional encoding dynamically based on actual sequence length
    encoder_embedding = PositionalEncodingLayer(
        max_length=prompt_sequence_length,
        embedding_dim=config.embedding_dim,
        name="add_pos_encoding_encoder",
    )(encoder_embedding)
    x = layers.Dropout(config.dropout_rate)(encoder_embedding)

    for i in range(config.num_layers):
        x = TransformerEncoderLayer(config, name=f"encoder_layer_{i}")(
            x, training=None, mask=encoder_padding_mask
        )
    encoder_output = x

    # Base decoder embedding from card IDs
    decoder_embedding = layers.Embedding(
        input_dim=deck_vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=False,
        name="deck_embedding",
    )(decoder_inputs)

    # Enhance decoder embedding with card features if enabled
    if use_card_features and card_cost_features is not None:
        card_cost_vals = ExtractFeatureLayer(name="extract_cost")(card_cost_features)
        card_power_vals = ExtractFeatureLayer(name="extract_power")(card_power_features)
        card_color_vals = ExtractFeatureLayer(name="extract_color")(card_color_features)
        card_type_vals = ExtractFeatureLayer(name="extract_type")(card_type_features)
        card_ability_vals = ExtractFeatureLayer(name="extract_ability")(card_ability_features)
        card_has_ability_vals = ExtractFeatureLayer(name="extract_has_ability")(card_has_ability_features)
        
        # Project card features to embedding space
        # Cost embedding (vocab_size -> embedding_dim)
        cost_embedding = layers.Embedding(
            input_dim=12,  # 0-10 cost + -1 for unknown
            output_dim=card_feature_dim // 6,
            name="cost_embedding",
        )(card_cost_vals)
        
        # Power embedding (vocab_size -> embedding_dim)
        power_embedding = layers.Embedding(
            input_dim=20001,  # 0-20000 power + -1 for unknown
            output_dim=card_feature_dim // 6,
            name="power_embedding",
        )(card_power_vals)
        
        # Color embedding (vocab_size -> embedding_dim)
        color_embedding = layers.Embedding(
            input_dim=8,  # 7 colors + 1 unknown
            output_dim=card_feature_dim // 6,
            name="color_embedding",
        )(card_color_vals)
        
        # Type embedding (vocab_size -> embedding_dim)
        type_embedding = layers.Embedding(
            input_dim=6,  # 5 types + 1 unknown
            output_dim=card_feature_dim // 6,
            name="type_embedding",
        )(card_type_vals)
        
        # Ability text embedding (vocab_size, 64 -> embedding_dim)
        # First embed the ability tokens, then pool them
        ability_token_embedding = layers.Embedding(
            input_dim=2000,  # Ability vocab size
            output_dim=card_feature_dim // 6,
            name="ability_token_embedding",
        )(card_ability_vals)  # Shape: (vocab_size, 64, dim)
        ability_embedding = layers.GlobalAveragePooling1D(name="ability_pooling")(
            ability_token_embedding
        )  # Shape: (vocab_size, dim)
        
        # Has ability flag embedding
        has_ability_embedding = layers.Embedding(
            input_dim=2,  # 0 or 1
            output_dim=card_feature_dim // 6,
            name="has_ability_embedding",
        )(card_has_ability_vals)
        
        # Concatenate all feature embeddings
        card_feature_embedding = layers.Concatenate(axis=-1, name="card_feature_concat")(
            [
                cost_embedding,
                power_embedding,
                color_embedding,
                type_embedding,
                ability_embedding,
                has_ability_embedding,
            ]
        )  # Shape: (vocab_size, card_feature_dim)
        
        # Project to match decoder embedding dimension
        card_feature_projection = layers.Dense(
            config.embedding_dim,
            name="card_feature_projection",
        )(card_feature_embedding)  # Shape: (vocab_size, embedding_dim)
        
        def gather_card_features(inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
            projection, token_ids = inputs  # projection -> (vocab_size, embed_dim), token_ids -> (batch, seq_len)
            token_ids = tf.cast(token_ids, tf.int32)
            flat_token_ids = tf.reshape(token_ids, [-1])
            gathered = tf.gather(projection, flat_token_ids, axis=0)
            seq_shape = tf.shape(token_ids)
            embed_dim = tf.shape(projection)[-1]
            gathered = tf.reshape(gathered, tf.concat([seq_shape, [embed_dim]], axis=0))
            return gathered

        gathered_features = layers.Lambda(
            gather_card_features,
            name="gather_card_features",
        )([card_feature_projection, decoder_inputs])

        decoder_embedding = layers.Add(name="combine_card_features")(
            [decoder_embedding, gathered_features]
        )
    else:
        # No card features - use base embedding only
        pass

    # Add positional encoding dynamically based on actual sequence length
    decoder_embedding = PositionalEncodingLayer(
        max_length=deck_sequence_length,
        embedding_dim=config.embedding_dim,
        name="add_pos_encoding_decoder",
    )(decoder_embedding)
    y = layers.Dropout(config.dropout_rate)(decoder_embedding)

    attention_weights = {}
    for i in range(config.num_layers):
        y, block1, block2 = TransformerDecoderLayer(config, name=f"decoder_layer_{i}")(
            y,
            enc_output=encoder_output,
            training=None,
            look_ahead_mask=combined_decoder_mask,
            padding_mask=encoder_padding_mask,
        )
        attention_weights[f"decoder_layer{i+1}_block1"] = block1
        attention_weights[f"decoder_layer{i+1}_block2"] = block2

    main_output = layers.Dense(deck_vocab_size, name="main")(y)
    pooled_decoder = layers.GlobalAveragePooling1D(name="decoder_global_pool")(y)
    type_aux_output = layers.Dense(len(TYPE_BUCKETS), activation="softmax", name="type_aux")(pooled_decoder)
    cost_aux_output = layers.Dense(COST_BUCKETS, activation="softmax", name="cost_aux")(pooled_decoder)
    freq_hist_output = layers.Dense(
        deck_vocab_size,
        activation="sigmoid",
        name="freq_hist",
    )(pooled_decoder)

    # Phase 1: Sequence-level output - compute frequency histogram from predicted tokens
    # This allows us to add a sequence-level loss that directly penalizes singleton-heavy generations
    from ..config import DeckConfig
    deck_config = DeckConfig()
    predicted_sequence_freq_hist = PredictedSequenceFreqHistLayer(
        vocab_size=deck_vocab_size,
        pad_token_id=pad_token_id,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        max_copies=deck_config.max_copies_per_card,
        name="predicted_sequence_freq_hist"
    )(main_output)

    # Build model inputs list
    model_inputs = [encoder_inputs, decoder_inputs]
    if use_card_features and card_cost_features is not None:
        model_inputs.extend([
            card_cost_features,
            card_power_features,
            card_color_features,
            card_type_features,
            card_ability_features,
            card_has_ability_features,
        ])

    model_outputs = {
        "main": main_output,
        "type_aux": type_aux_output,
        "cost_aux": cost_aux_output,
        "freq_hist": freq_hist_output,
        "predicted_sequence_freq_hist": predicted_sequence_freq_hist,
    }

    model = Model(inputs=model_inputs, outputs=model_outputs, name="deck_transformer")
    return model

