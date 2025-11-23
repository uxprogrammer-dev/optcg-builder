"""
TensorFlow input pipelines for prompt→deck training.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import random
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from ..config import DeckConfig, PromptConfig
from ..data import CardRecord, CardRepository, DeckSchema, PromptDeckExample
from .card_features import build_card_feature_matrix, create_ability_vectorizer
from .rules import load_rules_summary

PARAPHRASE_MAP = {
    "build": ["construct", "assemble", "put together"],
    "create": ["craft", "devise"],
    "design": ["plan", "engineer"],
    "deck": ["deck list", "lineup"],
    "strategy": ["gameplan", "approach"],
    "competitive": ["tournament-ready", "high-powered"],
}
TYPE_BUCKETS = ("CHARACTER", "EVENT", "STAGE", "OTHER")
COST_BUCKETS = 11
def _example_distributions(
    example: PromptDeckExample,
    repository: Optional[CardRepository],
) -> Tuple[List[float], List[float]]:
    if repository is None:
        type_uniform = [1.0 / len(TYPE_BUCKETS)] * len(TYPE_BUCKETS)
        cost_uniform = [1.0 / COST_BUCKETS] * COST_BUCKETS
        return type_uniform, cost_uniform

    type_counts: Counter[str] = Counter()
    cost_hist = [0] * COST_BUCKETS

    for card_id in example.deck.main_deck:
        record = repository.by_id(card_id)
        if record is None or record.is_leader:
            continue
        bucket = _normalize_type(record.type)
        type_counts[bucket] += 1
        cost = record.cost if record.cost is not None else 0
        cost_bucket = cost if 0 <= cost < COST_BUCKETS - 1 else COST_BUCKETS - 1
        cost_hist[cost_bucket] += 1

    total_types = sum(type_counts.values()) or 1
    type_distribution = [type_counts.get(bucket, 0) / total_types for bucket in TYPE_BUCKETS]
    total_cost = sum(cost_hist) or 1
    cost_distribution = [value / total_cost for value in cost_hist]
    return type_distribution, cost_distribution


def _format_control_tokens(
    type_distribution: Sequence[float],
    cost_distribution: Sequence[float],
) -> str:
    type_parts = [
        f"{bucket[:2]}={value:.2f}" for bucket, value in zip(TYPE_BUCKETS, type_distribution)
    ]
    low = sum(cost_distribution[:3])
    mid = sum(cost_distribution[3:6])
    high = sum(cost_distribution[6:])
    cost_parts = [f"L={low:.2f}", f"M={mid:.2f}", f"H={high:.2f}"]
    return f"[TYPE:{','.join(type_parts)}] [COST:{','.join(cost_parts)}]"


def _build_vectorizer(prompt_config: PromptConfig) -> TextVectorization:
    def _standardize(text: tf.Tensor) -> tf.Tensor:
        text = tf.strings.regex_replace(text, r"\[.*?\]", " ")
        if prompt_config.lowercase:
            text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, r"[^a-z0-9'\"!?.,:;\s-]", " ")
        return tf.strings.strip(text)

    vectorizer = TextVectorization(
        max_tokens=prompt_config.vocab_size,
        standardize=_standardize,
        output_mode="int",
        output_sequence_length=prompt_config.max_length,
    )
    return vectorizer


def create_text_vectorizer(
    prompts: Sequence[str],
    prompt_config: PromptConfig = PromptConfig(),
    batch_size: int = 128,
) -> TextVectorization:
    """
    Fit a TextVectorization layer on the supplied prompts.
    """

    vectorizer = _build_vectorizer(prompt_config)
    dataset = tf.data.Dataset.from_tensor_slices(list(prompts)).batch(batch_size)
    vectorizer.adapt(dataset)
    return vectorizer


def create_text_vectorizer_from_vocab(
    vocabulary: Sequence[str],
    prompt_config: PromptConfig = PromptConfig(),
) -> TextVectorization:
    vectorizer = _build_vectorizer(prompt_config)
    vectorizer.set_vocabulary(vocabulary)
    return vectorizer


def build_card_vocabulary(
    examples: Iterable[PromptDeckExample],
    deck_config: DeckConfig = DeckConfig(),
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a deterministic mapping of card IDs to integer token ids.
    """
    reserved_tokens = [
        deck_config.pad_token,
        deck_config.start_token,
        deck_config.end_token,
        deck_config.unknown_token,
    ]
    card_to_index: Dict[str, int] = {}
    index_to_card: Dict[int, str] = {}

    def _add_token(token: str) -> None:
        if token in card_to_index:
            return
        index = len(card_to_index)
        card_to_index[token] = index
        index_to_card[index] = token

    for token in reserved_tokens:
        _add_token(token)

    for example in examples:
        for token in _deck_tokens(example.deck):
            _add_token(token)

    return card_to_index, index_to_card


def _deck_tokens(deck: DeckSchema) -> List[str]:
    tokens = [deck.leader_id]
    tokens.extend(deck.main_deck)
    if deck.sideboard:
        tokens.extend(deck.sideboard)
    return tokens


def encode_deck_sequence(
    deck: DeckSchema,
    card_to_index: Dict[str, int],
    deck_config: DeckConfig = DeckConfig(),
) -> List[int]:
    tokens = _deck_tokens(deck)
    uk = card_to_index[deck_config.unknown_token]

    encoded = [card_to_index[deck_config.start_token]]
    encoded.extend(card_to_index.get(token, uk) for token in tokens[: deck_config.max_total_cards])
    encoded.append(card_to_index[deck_config.end_token])

    max_length = deck_config.max_total_cards + 2  # BOS + deck + EOS
    pad_value = card_to_index[deck_config.pad_token]
    if len(encoded) < max_length:
        encoded.extend([pad_value] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length]
        encoded[-1] = card_to_index[deck_config.end_token]
    return encoded


def decode_card_sequence(
    indices: Sequence[int],
    index_to_card: Dict[int, str],
    deck_config: DeckConfig = DeckConfig(),
) -> List[str]:
    tokens: List[str] = []
    special = {
        deck_config.pad_token,
        deck_config.start_token,
        deck_config.end_token,
    }
    for idx in indices:
        token = index_to_card.get(int(idx), deck_config.unknown_token)
        if token in special:
            continue
        tokens.append(token)
    return tokens


def _normalize_type(card_type: Optional[str]) -> str:
    if not card_type:
        return "OTHER"
    value = card_type.upper()
    if value in TYPE_BUCKETS:
        return value
    return "OTHER"


def make_tf_dataset(
    examples: Sequence[PromptDeckExample],
    prompt_vectorizer: TextVectorization,
    card_to_index: Dict[str, int],
    deck_config: DeckConfig = DeckConfig(),
    prompt_config: PromptConfig = PromptConfig(),
    batch_size: int = 32,
    shuffle_buffer: int = 512,
    seed: int = 123,
    cache: bool = True,
    prompt_dropout_rate: float = 0.1,
    prompt_paraphrase_rate: float = 0.2,
    deck_shuffle_rate: float = 0.05,  # Reduced from 0.2 - allow duplicates to appear consecutively so model learns realistic card counts
    shuffle: bool = True,
    data_root: Optional[Path] = None,
    repository: Optional[CardRepository] = None,
    use_card_features: bool = True,
    example_weights: Optional[Sequence[float]] = None,
    include_control_tokens: bool = True,
) -> Tuple[tf.data.Dataset, Optional[Dict[str, tf.Tensor]]]:
    """
    Construct a tf.data pipeline returning (features, targets).
    
    Returns:
        dataset: tf.data.Dataset with (inputs, targets) where inputs is a tuple
                 of (prompt_tokens, decoder_input) or (prompt_tokens, decoder_input, card_features)
        card_features: Optional dictionary of card feature tensors if use_card_features=True
    """
    
    # Build card feature matrix if requested
    card_features: Optional[Dict[str, tf.Tensor]] = None
    if use_card_features and repository:
        try:
            # Create ability vectorizer
            ability_vectorizer = create_ability_vectorizer(repository, vocab_size=2000, max_length=64)
            # Build feature matrix for all cards in vocabulary
            card_features = build_card_feature_matrix(
                card_to_index,
                repository,
                ability_vectorizer=ability_vectorizer,
                max_ability_length=64,
            )
        except Exception as e:
            # If card features fail to build, continue without them
            import warnings
            warnings.warn(f"Failed to build card features: {e}. Continuing without card features.")
            card_features = None
            use_card_features = False

    encoded_decks = [
        encode_deck_sequence(example.deck, card_to_index, deck_config=deck_config)
        for example in examples
    ]
    
    # Compute type/cost distributions and control tokens BEFORE building prompts
    type_labels = []
    cost_labels = []
    control_tokens_list: List[str] = []
    for example in examples:
        type_dist, cost_dist = _example_distributions(example, repository)
        type_labels.append(type_dist)
        cost_labels.append(cost_dist)
        control_tokens_list.append(
            _format_control_tokens(type_dist, cost_dist) if include_control_tokens else ""
        )
    
    # Optionally augment prompts with rules summary
    rules_summary = None
    if prompt_config.rules_augmentation_rate > 0.0:
        try:
            rules_summary = load_rules_summary(data_root=data_root)
        except Exception:
            # If rules loading fails, continue without rules augmentation
            pass
    
    prompts = []
    for idx, example in enumerate(examples):
        leader_context_lines = []
        ability_line = example.leader_ability.strip() if isinstance(example.leader_ability, str) else ""
        if ability_line:
            leader_context_lines.append(f"- Ability: {ability_line}")
        elif ability_line == "":
            leader_context_lines.append("- Ability: Not specified")

        subtypes = ", ".join(example.leader_subtypes or [])
        leader_context_lines.append(f"- Types: {subtypes or 'None'}")

        colors = ", ".join(example.leader_colors or [])
        leader_context_lines.append(f"- Colors: {colors or 'Unknown'}")

        leader_context = ""
        if leader_context_lines:
            leader_context = "Leader Context:\n" + "\n".join(leader_context_lines)

        prompt_body = example.prompt
        if leader_context:
            prompt_body = f"{leader_context}\n\n{prompt_body}"

        augmented_prompt = prompt_body
        if rules_summary and prompt_config.rules_augmentation_rate > 0.0:
            random.seed(seed + hash(example.prompt) % 1000)
            if random.random() < prompt_config.rules_augmentation_rate:
                augmented_prompt = f"{rules_summary}\n\n{prompt_body}"

        control_prefix = control_tokens_list[idx] if include_control_tokens else ""
        if control_prefix:
            augmented_prompt = f"{control_prefix} {augmented_prompt}"
        prompts.append(augmented_prompt)
    
    splits = [example.split or "train" for example in examples]

    weight_values = example_weights or [1.0] * len(examples)
    if len(weight_values) != len(examples):
        raise ValueError("example_weights length must match examples length.")
    weights_tensor = tf.constant(weight_values, dtype=tf.float32)

    type_tensor = tf.constant(type_labels, dtype=tf.float32)
    cost_tensor = tf.constant(cost_labels, dtype=tf.float32)

    prompt_ds = tf.data.Dataset.from_tensor_slices(prompts)
    deck_ds = tf.data.Dataset.from_tensor_slices(encoded_decks)
    split_ds = tf.data.Dataset.from_tensor_slices(splits)
    weight_ds = tf.data.Dataset.from_tensor_slices(weights_tensor)
    type_ds = tf.data.Dataset.from_tensor_slices(type_tensor)
    cost_ds = tf.data.Dataset.from_tensor_slices(cost_tensor)

    dataset = tf.data.Dataset.zip((prompt_ds, deck_ds, split_ds, weight_ds, type_ds, cost_ds))

    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    pad_value = tf.constant(card_to_index[deck_config.pad_token], dtype=tf.int32)
    bos_value = tf.constant(card_to_index[deck_config.start_token], dtype=tf.int32)
    eos_value = tf.constant(card_to_index[deck_config.end_token], dtype=tf.int32)

    vocab_size = len(card_to_index)
    max_copies = tf.constant(deck_config.max_copies_per_card, dtype=tf.float32)
    special_token_mask = tf.tensor_scatter_nd_update(
        tf.ones((vocab_size,), dtype=tf.float32),
        tf.constant(
            [
                [card_to_index[deck_config.pad_token]],
                [card_to_index[deck_config.start_token]],
                [card_to_index[deck_config.end_token]],
            ],
            dtype=tf.int32,
        ),
        tf.zeros((3,), dtype=tf.float32),
    )

    def _build_copy_ratio_targets(target_tokens: tf.Tensor) -> tf.Tensor:
        """
        Convert a sequence of deck tokens into normalized copy counts per card.

        Returns a vector where each index stores (#copies / max_copies_per_card),
        ensuring the model learns precise card multiplicities instead of binary flags.
        
        Example: If a card appears 2 times in the deck, its target value will be 2/4 = 0.5.
        This allows the model to learn that cards should appear in multiples (2x, 3x, 4x)
        rather than just being present (0/1).
        """
        token_one_hot = tf.one_hot(target_tokens, depth=vocab_size, dtype=tf.float32)
        copy_counts = tf.reduce_sum(token_one_hot, axis=0)
        copy_counts = copy_counts * special_token_mask
        capped_counts = tf.minimum(copy_counts, max_copies)
        return tf.math.divide_no_nan(capped_counts, max_copies)

    def _map_fn(
        prompt: tf.Tensor,
        deck_vector: tf.Tensor,
        split: tf.Tensor,
        weight: tf.Tensor,
        type_label: tf.Tensor,
        cost_label: tf.Tensor,
    ):
        # Note: card_features are constant and will be passed separately to the model
        augmented_prompt = prompt
        if prompt_paraphrase_rate > 0.0 and PARAPHRASE_MAP:
            for offset, (original, replacements) in enumerate(PARAPHRASE_MAP.items()):
                replacements_tensor = tf.constant(replacements, dtype=tf.string)
                random_draw = tf.random.uniform([], seed=seed + offset)
                replacement_idx = tf.random.uniform(
                    [],
                    maxval=len(replacements),
                    dtype=tf.int32,
                    seed=seed + offset + 1,
                )
                replacement = replacements_tensor[replacement_idx]
                pattern = rf"\b{original}\b"
                candidate = tf.strings.regex_replace(augmented_prompt, pattern, replacement)
                augmented_prompt = tf.where(
                    tf.less(random_draw, prompt_paraphrase_rate),
                    candidate,
                    augmented_prompt,
                )

        prompt_tokens = prompt_vectorizer(augmented_prompt)
        if prompt_dropout_rate > 0.0:
            keep_mask = tf.random.uniform(tf.shape(prompt_tokens), seed=seed) > prompt_dropout_rate
            prompt_tokens = tf.where(keep_mask, prompt_tokens, tf.zeros_like(prompt_tokens))

        if deck_shuffle_rate > 0.0:
            def _shuffle_deck() -> tf.Tensor:
                valid_length = tf.reduce_sum(
                    tf.cast(tf.not_equal(deck_vector, pad_value), tf.int32)
                )
                valid_length = tf.maximum(valid_length, 2)
                core = deck_vector[1 : valid_length - 1]
                shuffled = tf.random.shuffle(core)
                prefix = tf.reshape(bos_value, (1,))
                suffix = tf.reshape(eos_value, (1,))
                pad_count = tf.shape(deck_vector)[0] - valid_length
                tail = tf.fill((pad_count,), pad_value)
                return tf.concat([prefix, shuffled, suffix, tail], axis=0)

            random_draw = tf.random.uniform([], seed=seed)
            deck_vector = tf.cond(
                tf.less(random_draw, deck_shuffle_rate),
                true_fn=_shuffle_deck,
                false_fn=lambda: deck_vector,
            )

        # For teacher forcing: decoder_input is [BOS, ...tokens, PAD] and decoder_target is [...tokens, EOS, PAD]
        # Both need to be the same length as deck_vector for batching (deck_sequence_length)
        decoder_input = deck_vector[:-1]  # Remove last token (EOS or PAD)
        # Pad decoder_input back to original length
        decoder_input = tf.concat([decoder_input, tf.reshape(pad_value, (1,))], axis=0)
        decoder_target = deck_vector[1:]  # Remove first token (BOS)
        # Pad decoder_target to match length (deck_vector[1:] is one shorter)
        decoder_target = tf.concat([decoder_target, tf.reshape(pad_value, (1,))], axis=0)
        # Return inputs as a tuple (encoder_input, decoder_input) and target as a tensor
        # Card features are constant and will be passed separately to the model during training
        freq_hist = _build_copy_ratio_targets(decoder_target)

        return (
            (prompt_tokens, decoder_input),
            {
                "main": decoder_target,
                "type_aux": type_label,
                "cost_aux": cost_label,
                "freq_hist": freq_hist,
                # Phase 1: Sequence-level loss target (same as freq_hist)
                "predicted_sequence_freq_hist": freq_hist,
            },
            weight,
        )

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, card_features

