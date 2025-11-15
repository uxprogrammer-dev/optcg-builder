"""
Card feature encoding utilities for ML model training.

This module provides functions to encode card metadata (cost, power, color, type, ability text)
into numerical feature vectors that can be used as additional inputs to the transformer model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from ..data import CardRecord, CardRepository


# Color mapping: One Piece TCG colors
COLOR_TO_IDX: Dict[str, int] = {
    "RED": 0,
    "GREEN": 1,
    "BLUE": 2,
    "PURPLE": 3,
    "BLACK": 4,
    "YELLOW": 5,
    "WHITE": 6,
}
NUM_COLORS = len(COLOR_TO_IDX)
UNKNOWN_COLOR_IDX = NUM_COLORS  # Index for unknown/missing color


# Type mapping: One Piece TCG card types
TYPE_TO_IDX: Dict[str, int] = {
    "LEADER": 0,
    "CHARACTER": 1,
    "EVENT": 2,
    "STAGE": 3,
    "DON!!": 4,
}
NUM_TYPES = len(TYPE_TO_IDX)
UNKNOWN_TYPE_IDX = NUM_TYPES  # Index for unknown/missing type


def encode_color(color: Optional[str]) -> int:
    """Encode card color to integer index."""
    if not color:
        return UNKNOWN_COLOR_IDX
    color_upper = color.upper().strip()
    # Handle multi-color (e.g., "RED/GREEN")
    if "/" in color_upper:
        # For multi-color, use the first color
        color_upper = color_upper.split("/")[0].strip()
    return COLOR_TO_IDX.get(color_upper, UNKNOWN_COLOR_IDX)


def encode_type(card_type: Optional[str]) -> int:
    """Encode card type to integer index."""
    if not card_type:
        return UNKNOWN_TYPE_IDX
    type_upper = card_type.upper().strip()
    return TYPE_TO_IDX.get(type_upper, UNKNOWN_TYPE_IDX)


def encode_card_features(
    card: CardRecord,
    ability_vectorizer: Optional[TextVectorization] = None,
    max_ability_length: int = 64,
) -> Dict[str, tf.Tensor]:
    """
    Encode a card's metadata into feature tensors.
    
    Returns a dictionary with:
    - 'cost': int32 scalar (0-10, or -1 for unknown)
    - 'power': int32 scalar (0-20000, or -1 for unknown)
    - 'color': int32 scalar (0-6 for colors, 7 for unknown)
    - 'type': int32 scalar (0-4 for types, 5 for unknown)
    - 'ability_embedding': int32 vector of shape (max_ability_length,) (ability text vectorized)
    - 'has_ability': int32 scalar (1 if card has ability text, 0 otherwise)
    """
    # Cost: normalize to 0-10 range, use -1 for unknown
    cost_value = card.cost if card.cost is not None else -1
    cost_value = max(-1, min(10, cost_value))  # Clamp to reasonable range
    
    # Power: normalize to 0-20000 range, use -1 for unknown
    power_value = card.power if card.power is not None else -1
    power_value = max(-1, min(20000, power_value))  # Clamp to reasonable range
    
    # Color and type indices
    color_idx = encode_color(card.color)
    type_idx = encode_type(card.type)
    
    # Ability text: vectorize if vectorizer is provided
    ability_text = card.ability or ""
    has_ability = 1 if ability_text.strip() else 0
    
    # Filter out very short texts that might cause issues (e.g., just "-" or single punctuation)
    # After standardization, these might become empty strings
    if ability_vectorizer and ability_text and ability_text.strip() and len(ability_text.strip()) > 1:
        try:
            # Vectorize ability text - ensure it's a string tensor with batch dimension
            # TextVectorization expects a batch, so we wrap it in a list
            ability_tensor = tf.constant([ability_text], dtype=tf.string)
            ability_tokens = ability_vectorizer(ability_tensor)
            
            # TextVectorization should return shape (batch, sequence_length) = (1, max_length)
            # But handle edge cases where it might return empty or wrong shape
            if tf.size(ability_tokens) == 0:
                # Empty result - use zeros
                ability_tokens = tf.zeros((max_ability_length,), dtype=tf.int32)
            elif len(ability_tokens.shape) == 2:
                # Remove batch dimension: (1, max_length) -> (max_length,)
                ability_tokens = ability_tokens[0]
                # Ensure it's the right length
                if ability_tokens.shape[0] != max_ability_length:
                    ability_tokens = tf.reshape(ability_tokens, (max_ability_length,))
            elif len(ability_tokens.shape) == 1:
                # Already 1D, but might be wrong length
                current_length = tf.shape(ability_tokens)[0]
                if current_length == 0:
                    ability_tokens = tf.zeros((max_ability_length,), dtype=tf.int32)
                elif current_length < max_ability_length:
                    # Pad to max_ability_length
                    padding = max_ability_length - current_length
                    ability_tokens = tf.pad(ability_tokens, [[0, padding]], constant_values=0)
                elif current_length > max_ability_length:
                    # Truncate to max_ability_length
                    ability_tokens = ability_tokens[:max_ability_length]
                # Ensure final shape
                ability_tokens = tf.reshape(ability_tokens, (max_ability_length,))
            else:
                # Unexpected shape, use zeros
                ability_tokens = tf.zeros((max_ability_length,), dtype=tf.int32)
            
            # Final check: ensure shape is exactly (max_ability_length,)
            ability_tokens = tf.reshape(ability_tokens, (max_ability_length,))
        except Exception as e:
            # If vectorization fails, use empty embedding
            import warnings
            warnings.warn(f"Failed to vectorize ability text '{ability_text[:50]}...': {e}")
            ability_tokens = tf.zeros((max_ability_length,), dtype=tf.int32)
    else:
        # Empty ability embedding
        ability_tokens = tf.zeros((max_ability_length,), dtype=tf.int32)
    
    return {
        "cost": tf.constant(cost_value, dtype=tf.int32),
        "power": tf.constant(power_value, dtype=tf.int32),
        "color": tf.constant(color_idx, dtype=tf.int32),
        "type": tf.constant(type_idx, dtype=tf.int32),
        "ability_embedding": ability_tokens,
        "has_ability": tf.constant(has_ability, dtype=tf.int32),
    }


def build_card_feature_matrix(
    card_to_index: Dict[str, int],
    repository: CardRepository,
    ability_vectorizer: Optional[TextVectorization] = None,
    max_ability_length: int = 64,
) -> Dict[str, tf.Tensor]:
    """
    Build a feature matrix for all cards in the vocabulary.
    
    Returns a dictionary with feature tensors of shape (vocab_size, feature_dim):
    - 'cost': (vocab_size,)
    - 'power': (vocab_size,)
    - 'color': (vocab_size,)
    - 'type': (vocab_size,)
    - 'ability_embedding': (vocab_size, max_ability_length)
    - 'has_ability': (vocab_size,)
    """
    vocab_size = len(card_to_index)
    index_to_card = {idx: card_id for card_id, idx in card_to_index.items()}
    
    # Initialize feature arrays
    costs = []
    powers = []
    colors = []
    types = []
    ability_embeddings = []
    has_abilities = []
    
    # Build features for each card in vocabulary
    for idx in range(vocab_size):
        card_id = index_to_card.get(idx)
        if not card_id:
            # Unknown card - use default values
            costs.append(-1)
            powers.append(-1)
            colors.append(UNKNOWN_COLOR_IDX)
            types.append(UNKNOWN_TYPE_IDX)
            ability_embeddings.append(tf.zeros((max_ability_length,), dtype=tf.int32))
            has_abilities.append(0)
            continue
        
        card = repository.by_id(card_id)
        if not card:
            # Card not found - use default values
            costs.append(-1)
            powers.append(-1)
            colors.append(UNKNOWN_COLOR_IDX)
            types.append(UNKNOWN_TYPE_IDX)
            ability_embeddings.append(tf.zeros((max_ability_length,), dtype=tf.int32))
            has_abilities.append(0)
            continue
        
        # Encode card features
        try:
            features = encode_card_features(card, ability_vectorizer, max_ability_length)
            costs.append(int(features["cost"].numpy()))
            powers.append(int(features["power"].numpy()))
            colors.append(int(features["color"].numpy()))
            types.append(int(features["type"].numpy()))
            # Ensure ability_embedding has correct shape and dtype
            ability_emb = features["ability_embedding"]
            if len(ability_emb.shape) == 0 or tf.size(ability_emb) == 0:
                ability_emb = tf.zeros((max_ability_length,), dtype=tf.int32)
            elif ability_emb.shape[0] != max_ability_length:
                # Reshape if needed
                ability_emb = tf.reshape(ability_emb, (max_ability_length,))
            # Ensure dtype is int32 (might be int64 from numpy conversion)
            ability_emb = tf.cast(ability_emb, tf.int32)
            ability_embeddings.append(ability_emb)
            has_abilities.append(int(features["has_ability"].numpy()))
        except Exception as e:
            # If encoding fails for this card, use default values
            import warnings
            warnings.warn(f"Failed to encode features for card {card_id}: {e}")
            costs.append(-1)
            powers.append(-1)
            colors.append(UNKNOWN_COLOR_IDX)
            types.append(UNKNOWN_TYPE_IDX)
            ability_embeddings.append(tf.zeros((max_ability_length,), dtype=tf.int32))
            has_abilities.append(0)
    
    # Ensure all ability embeddings are int32 before stacking
    ability_embeddings_int32 = [tf.cast(emb, tf.int32) for emb in ability_embeddings]
    
    return {
        "cost": tf.constant(costs, dtype=tf.int32),
        "power": tf.constant(powers, dtype=tf.int32),
        "color": tf.constant(colors, dtype=tf.int32),
        "type": tf.constant(types, dtype=tf.int32),
        "ability_embedding": tf.stack(ability_embeddings_int32, axis=0),
        "has_ability": tf.constant(has_abilities, dtype=tf.int32),
    }


def create_ability_vectorizer(
    repository: CardRepository,
    vocab_size: int = 2000,
    max_length: int = 64,
) -> TextVectorization:
    """
    Create a TextVectorization layer for card ability text.
    
    This vectorizer is trained on all card ability texts in the repository.
    """
    # Collect all ability texts
    ability_texts = []
    for card in repository.cards.values():
        if card.ability and card.ability.strip():
            ability_texts.append(card.ability.strip())
    
    if not ability_texts:
        # No abilities found - return empty vectorizer
        vectorizer = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=max_length,
        )
        vectorizer.adapt([""])
        return vectorizer
    
    # Create and adapt vectorizer
    vectorizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_length,
        standardize="lower_and_strip_punctuation",
    )
    
    # Adapt on ability texts
    dataset = tf.data.Dataset.from_tensor_slices(ability_texts).batch(128)
    vectorizer.adapt(dataset)
    
    return vectorizer

