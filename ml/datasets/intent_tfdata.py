"""
TensorFlow input pipelines for intent classification training.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from ..config import PromptConfig
from ..data.intent_schema import IntentExample
from .tfdata import create_text_vectorizer

# Standard color order (must match model output)
COLORS = ["red", "blue", "green", "yellow", "purple", "black"]

# Strategy tags order
STRATEGIES = ["rush", "control", "combo", "midrange", "aggro"]


def build_intent_vocabularies(
    examples: List[IntentExample],
    leader_ids: List[str],
    keyword_vocab_size: int = 1000,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Build vocabularies for intent classification.
    
    Returns:
        Tuple of (color_to_index, index_to_color, keyword_to_index, index_to_keyword,
                  leader_to_index, index_to_leader, strategy_to_index, index_to_strategy)
    """
    # Color vocabulary (fixed order)
    color_to_index: Dict[str, int] = {color: i for i, color in enumerate(COLORS)}
    index_to_color: Dict[int, str] = {i: color for i, color in enumerate(COLORS)}

    # Keyword vocabulary (from examples)
    keyword_counter = Counter()
    for example in examples:
        for keyword in example.keywords:
            keyword_counter[keyword.lower().strip()] += 1

    # Take top N keywords
    top_keywords = [kw for kw, _ in keyword_counter.most_common(keyword_vocab_size)]
    keyword_to_index: Dict[str, int] = {kw: i for i, kw in enumerate(top_keywords)}
    index_to_keyword: Dict[int, str] = {i: kw for i, kw in enumerate(top_keywords)}

    # Leader vocabulary (from provided leader_ids)
    leader_to_index: Dict[str, int] = {lid.upper(): i for i, lid in enumerate(leader_ids)}
    index_to_leader: Dict[int, str] = {i: lid.upper() for i, lid in enumerate(leader_ids)}

    # Strategy vocabulary (fixed order)
    strategy_to_index: Dict[str, int] = {strat: i for i, strat in enumerate(STRATEGIES)}
    index_to_strategy: Dict[int, str] = {i: strat for i, strat in enumerate(STRATEGIES)}

    return (
        color_to_index,
        index_to_color,
        keyword_to_index,
        index_to_keyword,
        leader_to_index,
        index_to_leader,
        strategy_to_index,
        index_to_strategy,
    )


def make_intent_dataset(
    examples: Sequence[IntentExample],
    prompt_vectorizer: TextVectorization,
    color_to_index: Dict[str, int],
    keyword_to_index: Dict[str, int],
    leader_to_index: Dict[str, int],
    strategy_to_index: Dict[str, int],
    prompt_config: PromptConfig = PromptConfig(),
    batch_size: int = 32,
    shuffle_buffer: int = 512,
    seed: int = 123,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Construct a tf.data pipeline for intent classification.
    
    Returns:
        tf.data.Dataset with (prompt_tokens, targets) where targets is a dict:
        {
            "colors": (batch, num_colors) binary,
            "keywords": (batch, num_keywords) binary,
            "leaders": (batch, num_leaders) binary,
            "strategies": (batch, num_strategies) binary,
        }
    """
    # Vectorize prompts
    prompts = [ex.prompt for ex in examples]
    prompt_tensors = prompt_vectorizer(tf.constant(prompts))

    # Build multi-label targets
    num_colors = len(color_to_index)
    num_keywords = len(keyword_to_index)
    num_leaders = len(leader_to_index)
    num_strategies = len(strategy_to_index)

    color_targets = []
    keyword_targets = []
    leader_targets = []
    strategy_targets = []

    for example in examples:
        # Colors: binary vector
        color_vec = [0.0] * num_colors
        for color in example.colors:
            color_lower = color.lower().strip()
            if color_lower in color_to_index:
                color_vec[color_to_index[color_lower]] = 1.0
        color_targets.append(color_vec)

        # Keywords: binary vector (sparse)
        keyword_vec = [0.0] * num_keywords
        for keyword in example.keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in keyword_to_index:
                keyword_vec[keyword_to_index[keyword_lower]] = 1.0
        keyword_targets.append(keyword_vec)

        # Leaders: binary vector
        leader_vec = [0.0] * num_leaders
        for leader_id in example.leader_ids:
            leader_upper = leader_id.upper()
            if leader_upper in leader_to_index:
                leader_vec[leader_to_index[leader_upper]] = 1.0
        leader_targets.append(leader_vec)

        # Strategies: binary vector
        strategy_vec = [0.0] * num_strategies
        for strategy in example.strategy_tags:
            strategy_lower = strategy.lower().strip()
            if strategy_lower in strategy_to_index:
                strategy_vec[strategy_to_index[strategy_lower]] = 1.0
        strategy_targets.append(strategy_vec)

    # Convert to tensors
    color_tensor = tf.constant(color_targets, dtype=tf.float32)
    keyword_tensor = tf.constant(keyword_targets, dtype=tf.float32)
    leader_tensor = tf.constant(leader_targets, dtype=tf.float32)
    strategy_tensor = tf.constant(strategy_targets, dtype=tf.float32)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            prompt_tensors,
            {
                "colors": color_tensor,
                "keywords": keyword_tensor,
                "leaders": leader_tensor,
                "strategies": strategy_tensor,
            },
        )
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

