"""
Utility functions for decoding decks from the trained model.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import tensorflow as tf

from ..config import DeckConfig
from ..data import CardRecord, CardRepository
from ..datasets import decode_card_sequence

NEG_INF = -1e9
_SET_PREFIXES = {"OP", "EB", "ST", "SB", "SP", "PR", "P", "DP"}
_SET_PATTERN = re.compile(r"\b([A-Z]{2,3})[-\s]?(\d{1,2})\b")
_PROMPT_KEYWORD_WEIGHTS = {
    "blocker": 0.8,
    "rush": 0.8,
    "double attack": 0.7,
    "trigger": 0.6,
    "counter": 0.6,
    "search": 0.6,
    "draw": 0.5,
    "impel down": 0.8,
    "world government": 0.7,
    "marine": 0.7,
    "navy": 0.7,
    "imu": 1.0,
    "bounty": 0.5,
    "k.o.": 0.5,
    "destroy": 0.5,
    "discard": 0.5,
    "banish": 0.5,
}
TYPE_BUCKETS = ("CHARACTER", "EVENT", "STAGE", "OTHER")
TYPE_ID_TO_BUCKET = {
    1: "CHARACTER",
    2: "EVENT",
    3: "STAGE",
}


def _split_colors(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    cleaned = raw.replace("&", "/")
    return {part.strip().upper() for part in cleaned.split("/") if part.strip()}


def _normalize_card_set(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    core = code.upper().split("_")[0]
    prefix = core.split("-")[0]
    match = re.match(r"([A-Z]{2,3})(\d{1,2})", prefix)
    if not match:
        return None
    return f"{match.group(1)}{int(match.group(2)):02d}"


def _normalize_card_id_to_base(card_id: str) -> str:
    """
    Normalize a card ID to its base code (remove variant suffixes like _p1, _r1).
    
    Examples:
        OP01-017_p1 -> OP01-017
        OP01-017_r1 -> OP01-017
        OP01-017 -> OP01-017
    """
    if not card_id:
        return card_id
    # Split on underscore and take the first part (base code)
    return card_id.upper().split("_")[0]


def _extract_set_filters(prompt: str) -> Set[str]:
    filters: Set[str] = set()
    for prefix, number in _SET_PATTERN.findall(prompt.upper()):
        if prefix in _SET_PREFIXES:
            filters.add(f"{prefix}{int(number):02d}")
    return filters


def _resolve_record_for_token(
    token_id: int,
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
) -> Optional[CardRecord]:
    if repository is None:
        return None
    card_id = index_to_card.get(int(token_id))
    if not card_id:
        return None
    return repository.by_id(card_id)


def _build_penalty_vector(
    vocab_size: int,
    special_ids: Set[int],
    allow_fn,
) -> tf.Tensor:
    penalties = [0.0] * vocab_size
    for idx in range(vocab_size):
        if idx in special_ids:
            continue
        if not allow_fn(idx):
            penalties[idx] = NEG_INF
    return tf.constant(penalties, dtype=tf.float32)


def _build_color_penalty(
    allowed_colors: Set[str],
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
    special_ids: Set[int],
) -> Optional[tf.Tensor]:
    if not allowed_colors or repository is None:
        return None
    vocab_size = len(card_to_index)

    def allow(idx: int) -> bool:
        record = _resolve_record_for_token(idx, index_to_card, repository)
        if record is None:
            return True
        card_colors = _split_colors(record.color)
        if not card_colors:
            return True
        return bool(card_colors & allowed_colors)

    return _build_penalty_vector(vocab_size, special_ids, allow)


def _build_set_penalty(
    allowed_sets: Set[str],
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
    special_ids: Set[int],
) -> Optional[tf.Tensor]:
    if not allowed_sets or repository is None:
        return None
    vocab_size = len(card_to_index)

    def allow(idx: int) -> bool:
        record = _resolve_record_for_token(idx, index_to_card, repository)
        if record is None:
            return False
        card_set = _normalize_card_set(record.code)
        if card_set is None:
            return False
        return card_set in allowed_sets

    return _build_penalty_vector(vocab_size, special_ids, allow)


def _build_leader_penalty(
    card_features: Optional[Dict[str, tf.Tensor]],
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
    special_ids: Set[int],
) -> Optional[tf.Tensor]:
    vocab_size = len(card_to_index)

    if card_features and "type" in card_features:
        type_values = card_features["type"]
        if hasattr(type_values, "numpy"):
            type_array = type_values.numpy()
        else:
            type_array = tf.cast(type_values, tf.int32).numpy()

        def allow(idx: int) -> bool:
            if idx < len(type_array):
                return int(type_array[idx]) != 0
            record = _resolve_record_for_token(idx, index_to_card, repository)
            return not (record and record.is_leader)

        return _build_penalty_vector(vocab_size, special_ids, allow)

    if repository is None:
        return None

    def allow(idx: int) -> bool:
        record = _resolve_record_for_token(idx, index_to_card, repository)
        return not (record and record.is_leader)

    return _build_penalty_vector(vocab_size, special_ids, allow)


def _build_cost_restriction_penalty(
    leader_record: Optional[CardRecord],
    card_features: Optional[Dict[str, tf.Tensor]],
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
    special_ids: Set[int],
) -> Optional[tf.Tensor]:
    """
    Build penalty for cards that violate leader cost restrictions.
    
    Parses leader ability text for restrictions like:
    - "cannot include cards with a cost of 5 or more"
    - "cannot include cards with cost X or more"
    """
    if not leader_record or not repository:
        return None
    
    ability_text = (leader_record.ability or "").lower()
    max_cost = None
    
    # Pattern: "cannot include cards with a cost of X or more"
    # Pattern: "cannot include cards with cost X or more"
    max_cost_pattern = re.compile(
        r'cannot\s+include\s+cards?\s+with\s+(?:a\s+)?cost\s+of\s+(\d+)\s+or\s+more',
        re.IGNORECASE
    )
    match = max_cost_pattern.search(ability_text)
    if match:
        max_cost = int(match.group(1)) - 1  # If "5 or more" banned, max allowed is 4
    
    # Pattern: "cannot include cards with cost X or less" (less common)
    min_cost_pattern = re.compile(
        r'cannot\s+include\s+cards?\s+with\s+(?:a\s+)?cost\s+of\s+(\d+)\s+or\s+less',
        re.IGNORECASE
    )
    match = min_cost_pattern.search(ability_text)
    if match:
        min_cost = int(match.group(1)) + 1  # If "2 or less" banned, min allowed is 3
        # For now, we only handle max_cost (most common case)
        # min_cost can be added later if needed
    
    if max_cost is None:
        return None  # No cost restriction found
    
    vocab_size = len(card_to_index)
    
    def allow(idx: int) -> bool:
        if idx in special_ids:
            return True
        record = _resolve_record_for_token(idx, index_to_card, repository)
        if not record:
            return True  # Allow unknown cards (will be filtered later)
        card_cost = record.cost if record.cost is not None else 0
        return card_cost <= max_cost
    
    return _build_penalty_vector(vocab_size, special_ids, allow)


def _apply_penalty(logits: tf.Tensor, penalty: Optional[tf.Tensor]) -> tf.Tensor:
    if penalty is None:
        return logits
    return logits + penalty


def _apply_copy_limit_penalty(
    logits: tf.Tensor,
    copy_counts: Counter,
    max_copies: int,
    special_ids: Set[int],
) -> tf.Tensor:
    blocked = [
        token_id
        for token_id, count in copy_counts.items()
        if count >= max_copies and token_id not in special_ids
    ]
    if not blocked:
        return logits
    indices = tf.expand_dims(tf.constant(blocked, dtype=tf.int32), 1)
    penalties = tf.constant([NEG_INF] * len(blocked), dtype=logits.dtype)
    return tf.tensor_scatter_nd_add(logits, indices, penalties)


def _apply_repetition_bonus(
    logits: tf.Tensor,
    copy_counts: Counter,
    special_ids: Set[int],
    bonus: float = 0.5,  # Bonus per existing copy
) -> tf.Tensor:
    """Add bonus to cards that are already in the deck to encourage repetition."""
    vocab_size = logits.shape[0]
    bonus_vector = [0.0] * vocab_size
    
    for token_id, count in copy_counts.items():
        if token_id not in special_ids and count > 0 and count < 4:
            # Bonus increases with count (up to 3 copies), then stops at 4
            bonus_vector[token_id] = bonus * count
    
    if any(bonus_vector):
        return logits + tf.constant(bonus_vector, dtype=logits.dtype)
    return logits


def _apply_synergy_bonus(
    logits: tf.Tensor,
    generated: List[int],
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
    special_ids: Set[int],
    bonus: float = 0.3,
) -> tf.Tensor:
    """Add bonus to cards that synergize with already-generated cards."""
    if not repository or len(generated) < 3:  # Need at least leader + 1 card
        return logits
    
    vocab_size = logits.shape[0]
    bonus_vector = [0.0] * vocab_size
    
    # Get already-generated cards (skip leader and special tokens)
    generated_cards = [
        index_to_card.get(token_id) 
        for token_id in generated[2:]  # Skip BOS and leader
        if token_id not in special_ids and token_id in index_to_card
    ]
    
    if not generated_cards:
        return logits
    
    # Check each candidate card for synergy with generated cards
    for idx in range(vocab_size):
        if idx in special_ids:
            continue
        candidate_id = index_to_card.get(idx)
        if not candidate_id:
            continue
        
        candidate_record = repository.by_id(candidate_id)
        if not candidate_record:
            continue
        
        candidate_text = _collect_card_text(candidate_record).lower()
        
        # Check if candidate synergizes with any generated card
        for gen_card_id in generated_cards:
            gen_record = repository.by_id(gen_card_id)
            if not gen_record:
                continue
            
            gen_name = (gen_record.name or "").lower()
            gen_subtypes = set((gen_record.family or "").lower().split("/"))
            gen_text = _collect_card_text(gen_record).lower()
            
            # Check for name references
            if gen_name and gen_name.split()[0] in candidate_text:
                bonus_vector[idx] += bonus
                break
            
            # Check for subtype references
            for subtype in gen_subtypes:
                if subtype and len(subtype) > 3 and subtype in candidate_text:
                    bonus_vector[idx] += bonus
                    break
    
    if any(bonus_vector):
        return logits + tf.constant(bonus_vector, dtype=logits.dtype)
    return logits


def _collect_card_text(record: Optional[CardRecord]) -> str:
    if record is None:
        return ""
    parts = [
        record.name or "",
        record.family or "",
        record.ability or "",
        record.trigger or "",
        record.color or "",
        record.type or "",
    ]
    return " ".join(parts).lower()


def _build_prompt_bias_vector(
    prompt: str,
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
    special_ids: Set[int],
) -> Optional[tf.Tensor]:
    if repository is None:
        return None

    prompt_lower = prompt.lower()
    active_keywords = {
        keyword
        for keyword in _PROMPT_KEYWORD_WEIGHTS
        if keyword in prompt_lower
    }
    if not active_keywords:
        return None

    vocab_size = len(card_to_index)
    bias = [0.0] * vocab_size
    weight_map = {kw: _PROMPT_KEYWORD_WEIGHTS[kw] for kw in active_keywords}

    for idx in range(vocab_size):
        if idx in special_ids:
            continue
        record = _resolve_record_for_token(idx, index_to_card, repository)
        if record is None:
            continue
        text = _collect_card_text(record)
        for keyword, weight in weight_map.items():
            if keyword in text:
                bias[idx] += weight

    return tf.constant(bias, dtype=tf.float32)


def _normalize_card_type(card_type: Optional[str]) -> str:
    if not card_type:
        return "OTHER"
    value = card_type.upper()
    if value in TYPE_BUCKETS:
        return value
    return "OTHER"


def _build_token_type_lookup(
    card_features: Optional[Dict[str, tf.Tensor]],
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    repository: Optional[CardRepository],
) -> List[str]:
    vocab_size = len(card_to_index)
    lookup: List[str] = ["OTHER"] * vocab_size

    if card_features and "type" in card_features:
        type_tensor = card_features["type"]
        if hasattr(type_tensor, "numpy"):
            type_array = type_tensor.numpy()
        else:
            type_array = tf.cast(type_tensor, tf.int32).numpy()
        limit = min(len(type_array), vocab_size)
        for idx in range(limit):
            type_id = int(type_array[idx])
            if type_id == 0:
                lookup[idx] = "LEADER"
            else:
                lookup[idx] = TYPE_ID_TO_BUCKET.get(type_id, "OTHER")
    elif repository:
        for idx in range(vocab_size):
            record = _resolve_record_for_token(idx, index_to_card, repository)
            if record is None:
                continue
            lookup[idx] = "LEADER" if record.is_leader else _normalize_card_type(record.type)

    return lookup


def _lookup_leader_prior(
    leader_token_id: int,
    index_to_card: Dict[int, str],
    priors: Optional[Dict[str, Dict[str, object]]],
) -> Optional[Dict[str, object]]:
    if not priors:
        return None
    leader_card_id = index_to_card.get(int(leader_token_id))
    if not leader_card_id:
        return None
    if leader_card_id in priors:
        return priors[leader_card_id]
    base_id = leader_card_id.split("_")[0]
    return priors.get(base_id)


def _compute_type_targets(
    prior_data: Optional[Dict[str, object]],
    deck_size: int,
) -> Optional[Dict[str, int]]:
    if not prior_data:
        return None
    ratios = prior_data.get("avg_type_ratio")
    if not isinstance(ratios, Mapping):
        return None

    counts: Dict[str, int] = {}
    for bucket in TYPE_BUCKETS:
        ratio = float(ratios.get(bucket, 0.0))
        counts[bucket] = max(0, int(round(ratio * deck_size)))

    total = sum(counts.values())
    diff = deck_size - total

    if diff > 0:
        order = sorted(TYPE_BUCKETS, key=lambda bucket: float(ratios.get(bucket, 0.0)), reverse=True)
        index = 0
        while diff > 0 and order:
            bucket = order[index % len(order)]
            counts[bucket] += 1
            diff -= 1
            index += 1
    elif diff < 0:
        order = sorted(TYPE_BUCKETS, key=lambda bucket: float(ratios.get(bucket, 0.0)))
        index = 0
        while diff < 0 and order:
            bucket = order[index % len(order)]
            if counts[bucket] > 0:
                counts[bucket] -= 1
                diff += 1
            index += 1

    return counts


def _scatter_penalty(
    logits: tf.Tensor,
    indices: List[int],
    penalty_value: float,
) -> tf.Tensor:
    if not indices:
        return logits
    tensor_indices = tf.expand_dims(tf.constant(indices, dtype=tf.int32), 1)
    penalties = tf.constant([penalty_value] * len(indices), dtype=logits.dtype)
    return tf.tensor_scatter_nd_add(logits, tensor_indices, penalties)


def _apply_type_prior_penalty(
    logits: tf.Tensor,
    token_types: Sequence[str],
    type_counts: Counter[str],
    type_targets: Optional[Dict[str, int]],
    special_ids: Set[int],
) -> tf.Tensor:
    if not type_targets:
        return logits

    hard_block: List[int] = []
    soft_penalty: List[int] = []

    for idx, category in enumerate(token_types):
        if idx in special_ids or category == "LEADER":
            continue
        if category not in type_targets:
            continue
        target = type_targets[category]
        current = type_counts.get(category, 0)
        if current >= target + 1:
            hard_block.append(idx)
        elif current >= target and target > 0:
            soft_penalty.append(idx)

    logits = _scatter_penalty(logits, hard_block, NEG_INF)
    logits = _scatter_penalty(logits, soft_penalty, -2.0)
    return logits


def _extract_main_logits(outputs: tf.Tensor | Dict[str, tf.Tensor]) -> tf.Tensor:
    if isinstance(outputs, dict):
        main = outputs.get("main")
        if main is not None:
            return main
        # Fallback to first value in dictionary order
        return next(iter(outputs.values()))
    return outputs


def _extract_freq_hist(outputs: tf.Tensor | Dict[str, tf.Tensor]) -> Optional[tf.Tensor]:
    """Extract the frequency histogram output from model outputs."""
    if isinstance(outputs, dict):
        return outputs.get("freq_hist")
    return None


def _apply_duplicate_encouragement_bias(
    logits: tf.Tensor,
    copy_counts: Counter,
    special_ids: Set[int],
    index_to_card: Optional[Dict[int, str]] = None,  # NEW: Needed to normalize card IDs
     bias_strength: float = 50.0,  # Dramatically increased from 25.0 - model still generating all 1x cards
    unseen_penalty: float = 20.0,  # NEW: Penalize cards that haven't been generated yet
    min_cards_before_penalty: int = 15,  # NEW: Only penalize unseen cards after generating this many cards
) -> tf.Tensor:
    """
    Encourage duplicates by boosting cards that have been generated 1-3 times,
    and penalizing cards that haven't been generated yet (after generating some cards).
    
    Args:
        logits: Raw logits [vocab_size]
        copy_counts: Counter of how many times each card has been generated
        special_ids: Set of special token IDs to exclude
        bias_strength: How strongly to boost duplicates (higher = more aggressive)
        unseen_penalty: How strongly to penalize unseen cards (higher = more aggressive)
        min_cards_before_penalty: Only apply unseen penalty after generating this many cards
    
    Returns:
        Modified logits with duplicate encouragement bias applied
    """
    bias = tf.zeros_like(logits)
    vocab_size = tf.shape(logits)[0]
    vocab_size_int = int(vocab_size.numpy()) if hasattr(vocab_size, "numpy") else int(vocab_size)
    
    # Count total cards generated so far (excluding special tokens)
    # Since copy_counts now uses normalized base codes, we need to count unique base codes
    if index_to_card:
        unique_base_codes = set()
        for token_id, count in copy_counts.items():
            if token_id not in special_ids and count > 0:
                card_id = index_to_card.get(token_id, "")
                if card_id:
                    base_code = _normalize_card_id_to_base(card_id)
                    unique_base_codes.add(base_code)
        total_cards_generated = len(unique_base_codes)
    else:
        # Fallback: count all tokens (may overcount variants)
        total_cards_generated = sum(1 for token_id, count in copy_counts.items() if token_id not in special_ids and count > 0)
    
    # Only penalize unseen cards after we've generated a minimum number of cards
    # This allows the model to generate a diverse initial set, then forces duplicates
    if total_cards_generated >= min_cards_before_penalty:
        # Progressive penalty: stronger as we generate more cards
        # After 15 cards: full penalty
        # After 30 cards: 1.5x penalty
        # After 40 cards: 2x penalty
        penalty_multiplier = 1.0
        if total_cards_generated >= 40:
            penalty_multiplier = 2.0
        elif total_cards_generated >= 30:
            penalty_multiplier = 1.5
        
        for token_id in range(vocab_size_int):
            if token_id in special_ids:
                continue
            if token_id not in copy_counts or copy_counts[token_id] == 0:
                # Penalize unseen cards to encourage duplicates
                # Use a much stronger penalty that scales with the number of cards generated
                # After 15 cards: base penalty
                # After 30 cards: 2x penalty  
                # After 40 cards: 3x penalty
                base_penalty = unseen_penalty * penalty_multiplier
                # Additional scaling based on how many cards we've generated
                if total_cards_generated >= 40:
                    base_penalty *= 3.0
                elif total_cards_generated >= 30:
                    base_penalty *= 2.0
                penalty = -base_penalty
                bias = tf.tensor_scatter_nd_update(
                    bias,
                    [[token_id]],
                    [tf.constant(penalty, dtype=tf.float32)]
                )
    
    # Then, boost cards that have been generated 1-3 times (encourage 2x, 3x, 4x)
    # Cards with 1 copy: moderate boost (encourage 2x)
    # Cards with 2 copies: strong boost (encourage 3x)
    # Cards with 3 copies: very strong boost (encourage 4x)
    # Cards with 4+ copies: no boost (already at limit)
    
    # Group token IDs by base code to boost all variants together
    base_code_to_tokens: Dict[str, List[int]] = {}
    if index_to_card:
        vocab_size_int = int(vocab_size.numpy()) if hasattr(vocab_size, "numpy") else int(vocab_size)
        for token_id in range(vocab_size_int):
            if token_id in special_ids:
                continue
            card_id = index_to_card.get(token_id, "")
            if card_id:
                base_code = _normalize_card_id_to_base(card_id)
                if base_code not in base_code_to_tokens:
                    base_code_to_tokens[base_code] = []
                base_code_to_tokens[base_code].append(token_id)
    
    # Build base_code_counts from copy_counts to get accurate counts per base code
    # This ensures we boost ALL variants, not just the ones in copy_counts
    base_code_counts_from_copy: Dict[str, int] = {}
    for token_id, count in copy_counts.items():
        if token_id in special_ids:
            continue
        if index_to_card:
            card_id = index_to_card.get(token_id, "")
            if card_id:
                base_code = _normalize_card_id_to_base(card_id)
                # Use the maximum count for this base code (should be the same for all variants)
                base_code_counts_from_copy[base_code] = max(base_code_counts_from_copy.get(base_code, 0), count)
    
    # Boost ALL token IDs that map to base codes with count 1-3
    # This ensures ALL variants get boosted, not just the ones in copy_counts
    boosted_cards = []  # For debugging
    for base_code, count in base_code_counts_from_copy.items():
        if count == 1:
            # Encourage 2x copies - MUCH stronger boost for first duplicate
            # This is critical - we need to overcome the model's strong preference for diversity
            boost = bias_strength * 3.0  # Increased from 2.0 - need to make duplicates much more likely
        elif count == 2:
            # Encourage 3x copies
            boost = bias_strength * 3.5  # Increased from 2.5
        elif count == 3:
            # Encourage 4x copies
            boost = bias_strength * 4.0  # Increased from 3.0
        else:
            # Already at 4x or more, don't boost
            continue
        
        # Boost ALL token IDs that map to this base code (all variants)
        if index_to_card and base_code in base_code_to_tokens:
            variant_tokens = base_code_to_tokens[base_code]
            for variant_token_id in variant_tokens:
                if variant_token_id not in special_ids:
                    card_name = index_to_card.get(variant_token_id, f"<UNK:{variant_token_id}>")
                    boosted_cards.append((card_name, count, boost))
                    bias = tf.tensor_scatter_nd_update(
                        bias,
                        [[variant_token_id]],
                        [tf.constant(boost, dtype=tf.float32)]
                    )
    
    # Debug: show which cards are being boosted and their logit values
    if boosted_cards:
        import sys
        # Show logits for boosted cards (before and after boost)
        boosted_info = []
        for card_name, count, boost in boosted_cards[:10]:  # Show first 10
            # Find token_id for this card
            token_id = None
            for tid, cid in index_to_card.items():
                if cid == card_name:
                    token_id = tid
                    break
            if token_id is not None and token_id < vocab_size_int:
                orig_logit = float(logits[token_id].numpy()) if hasattr(logits[token_id], "numpy") else float(logits[token_id])
                boosted_logit = orig_logit + boost
                boosted_info.append(f"{card_name}({count}x): {orig_logit:.2f} -> {boosted_logit:.2f} (+{boost:.1f})")
        if boosted_info:
            print(f"DEBUG: Boosting {len(boosted_cards)} card variants. Sample logits: {', '.join(boosted_info[:5])}", file=sys.stderr)
        else:
            print(f"DEBUG: Boosting {len(boosted_cards)} card variants for duplicates: {boosted_cards[:5]}...", file=sys.stderr)
    
    return logits + bias


def _apply_freq_hist_bias(
    logits: tf.Tensor,
    freq_hist: Optional[tf.Tensor],
    copy_counts: Counter,
    special_ids: Set[int],
    bias_strength: float = 1.0,
) -> tf.Tensor:
    """
    Apply frequency histogram bias to encourage realistic card counts.
    
    Args:
        logits: Current token logits [vocab_size]
        freq_hist: Frequency histogram from model [vocab_size] (values 0-1 corresponding to expected copy ratio)
        copy_counts: Counter of how many times each card has been generated
        special_ids: Set of special token IDs to exclude
        bias_strength: How strongly to apply the bias (higher = more aggressive)
    
    Returns:
        Biased logits
    """
    if freq_hist is None:
        return logits
    
    # freq_hist shape is [vocab_size] (1D tensor from pooled decoder output)
    # Values are scaled between 0 and 1, where 1.0 ~= max_copies_per_card (4 copies)
    # We want to boost cards that:
    # 1. Have high freq_hist probability (should appear multiple times)
    # 2. Have already been generated but haven't reached their expected count yet
    
    # Flatten freq_hist to 1D if needed
    freq_hist_flat = tf.reshape(freq_hist, [-1])
    vocab_size = tf.shape(logits)[0]
    bias = tf.zeros_like(logits)
    
    # Convert to numpy for easier iteration (freq_hist is small, vocab_size)
    try:
        freq_hist_np = freq_hist_flat.numpy()
    except (AttributeError, TypeError):
        # Fallback: use tensor operations (slower but works)
        freq_hist_np = None
    
    if freq_hist_np is not None:
        # Skip if histogram provides no strong signal
        max_prob = float(max(freq_hist_np))
        if max_prob < 0.05:
            import logging
            logging.debug(
                "freq_hist peak < 0.05 (%.6f), skipping freq_hist bias",
                max_prob,
            )
            return logits
        
        # Use numpy for faster iteration
        for token_id, current_count in copy_counts.items():
            if token_id in special_ids or token_id >= len(freq_hist_np):
                continue
            # Expected count based on freq_hist (scaled by max copies)
            expected_prob = float(freq_hist_np[token_id])
            expected_count = expected_prob * 4.0  # Convert back to copies
            
            # If we haven't reached expected count, boost this card
            if current_count < expected_count:
                # Boost strength depends on how far we are from expected count
                # Cap boost to prevent extreme values that break generation
                boost = min((expected_count - current_count) * bias_strength, 12.0)
                bias = tf.tensor_scatter_nd_update(
                    bias,
                    [[token_id]],
                    [tf.constant(boost, dtype=tf.float32)]
                )
        
        # Also boost cards with high freq_hist that haven't been generated yet
        vocab_size_int = int(vocab_size.numpy()) if hasattr(vocab_size, "numpy") else int(vocab_size)
        for token_id in range(min(vocab_size_int, len(freq_hist_np))):
            if token_id in special_ids or token_id in copy_counts:
                continue
            prob = float(freq_hist_np[token_id])
            if prob > 0.25:  # Model expects at least one copy
                boost = min(prob * bias_strength * 4.0, 10.0)
                bias = tf.tensor_scatter_nd_update(
                    bias,
                    [[token_id]],
                    [tf.constant(boost, dtype=tf.float32)],
                )
    else:
        # Fallback: use tensor operations (slower)
        # Boost cards that have high freq_hist but low current count
        for token_id, current_count in copy_counts.items():
            if token_id in special_ids:
                continue
            # Expected count based on freq_hist (scaled by max copies)
            expected_prob = tf.gather(freq_hist_flat, token_id)
            expected_count = expected_prob * 4.0
            
            # If we haven't reached expected count, boost this card
            boost = tf.maximum(
                0.0,
                (expected_count - tf.cast(current_count, tf.float32)) * bias_strength,
            )
            bias = tf.tensor_scatter_nd_update(
                bias,
                [[token_id]],
                [boost]
            )
        
        # Also boost cards with high freq_hist that haven't been generated yet
        # This is more expensive with tensors, so we'll skip it in fallback mode
    
    return logits + bias


def _prepare_prompt_tensor(prompt: str, prompt_vectorizer) -> tf.Tensor:
    prompt_tensor = prompt_vectorizer(tf.constant([prompt]))
    return prompt_tensor


def _prepare_decoder_input(
    generated: Sequence[int],
    pad_token_id: int,
    max_length: int,
) -> tf.Tensor:
    padded = list(generated) + [pad_token_id] * (max_length - len(generated))
    return tf.constant([padded], dtype=tf.int32)


def greedy_generate(
    prompt: str,
    model: tf.keras.Model,
    prompt_vectorizer,
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    deck_config: DeckConfig = DeckConfig(),
    initial_sequence: Optional[List[int]] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_penalty: float = 0.0,
    card_features: Optional[Dict[str, tf.Tensor]] = None,
    use_card_features: bool = True,
    repository: Optional[CardRepository] = None,
    tournament_priors: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[str]:
    import sys
    prompt_tensor = _prepare_prompt_tensor(prompt, prompt_vectorizer)
    start_id = card_to_index[deck_config.start_token]
    end_id = card_to_index[deck_config.end_token]
    pad_id = card_to_index[deck_config.pad_token]
    max_length = deck_config.max_total_cards + 2

    # If initial_sequence is provided (e.g., [start_id, leader_id]), use it; otherwise start with just [start_id]
    generated = initial_sequence if initial_sequence is not None else [start_id]
    
    # Prepare card features if available and model supports them (these are constant across all steps)
    card_feature_inputs = []
    if card_features and use_card_features:
        # Card features are constant tensors, add them to model inputs
        # Expand dims to add batch dimension if needed (they should be (vocab_size,) or (1, vocab_size))
        cost_feat = tf.expand_dims(card_features["cost"], 0) if len(card_features["cost"].shape) == 1 else card_features["cost"]
        power_feat = tf.expand_dims(card_features["power"], 0) if len(card_features["power"].shape) == 1 else card_features["power"]
        color_feat = tf.expand_dims(card_features["color"], 0) if len(card_features["color"].shape) == 1 else card_features["color"]
        type_feat = tf.expand_dims(card_features["type"], 0) if len(card_features["type"].shape) == 1 else card_features["type"]
        ability_feat = tf.expand_dims(card_features["ability_embedding"], 0) if len(card_features["ability_embedding"].shape) == 2 else card_features["ability_embedding"]
        has_ability_feat = tf.expand_dims(card_features["has_ability"], 0) if len(card_features["has_ability"].shape) == 1 else card_features["has_ability"]
        card_feature_inputs = [cost_feat, power_feat, color_feat, type_feat, ability_feat, has_ability_feat]

    special_ids: Set[int] = {start_id, end_id, pad_id}
    # Build copy_counts using normalized card IDs (base codes without variants)
    # This ensures variant cards (OP01-017_p1, OP01-017_r1) are counted as the same card
    base_code_counts: Dict[str, int] = {}
    for token_id in generated:
        if token_id not in special_ids:
            card_id = index_to_card.get(token_id, "")
            if card_id:
                base_code = _normalize_card_id_to_base(card_id)
                base_code_counts[base_code] = base_code_counts.get(base_code, 0) + 1
    
    # Convert back to token_id-based Counter for compatibility with existing code
    # Map each token_id to the count of its base code
    copy_counts = Counter()
    for token_id in generated:
        if token_id not in special_ids:
            card_id = index_to_card.get(token_id, "")
            if card_id:
                base_code = _normalize_card_id_to_base(card_id)
                copy_counts[token_id] = base_code_counts.get(base_code, 0)

    set_filters = _extract_set_filters(prompt)
    set_penalty = _build_set_penalty(
        set_filters,
        card_to_index,
        index_to_card,
        repository,
        special_ids,
    )
    leader_penalty = _build_leader_penalty(
        card_features,
        card_to_index,
        index_to_card,
        repository,
        special_ids,
    )
    prompt_bias = _build_prompt_bias_vector(
        prompt,
        card_to_index,
        index_to_card,
        repository,
        special_ids,
    )
    token_types = _build_token_type_lookup(
        card_features,
        card_to_index,
        index_to_card,
        repository,
    )
    type_counts = Counter(
        token_types[token_id]
        for token_id in generated
        if token_id not in special_ids and token_id < len(token_types)
    )
    type_targets: Optional[Dict[str, int]] = None
    color_penalty_cache: Dict[int, Optional[tf.Tensor]] = {}
    cost_restriction_penalty_cache: Dict[int, Optional[tf.Tensor]] = {}
    leader_record: Optional[CardRecord] = None

    if repository and len(generated) >= 2:
        leader_record = _resolve_record_for_token(
            generated[1], index_to_card, repository
        )
        if leader_record:
            allowed_colors = _split_colors(leader_record.color)
            color_penalty_cache[generated[1]] = _build_color_penalty(
                allowed_colors,
                card_to_index,
                index_to_card,
                repository,
                special_ids,
            )
            # Also build cost restriction penalty
            cost_restriction_penalty_cache[generated[1]] = _build_cost_restriction_penalty(
                leader_record,
                card_features,
                card_to_index,
                index_to_card,
                repository,
                special_ids,
            )

    # Start generating from where we left off
    start_step = len(generated) - 1
    for step in range(start_step, max_length - 1):
        decoder_input = _prepare_decoder_input(generated, pad_id, max_length)
        # Rebuild copy_counts from full generated sequence at the start of each step
        # This ensures counts are accurate before applying biasing
        if len(generated) >= 2:  # Only rebuild if we have at least the leader
            base_code_counts: Dict[str, int] = {}
            for token_id in generated:
                if token_id not in special_ids:
                    card_id = index_to_card.get(token_id, "")
                    if card_id:
                        base_code = _normalize_card_id_to_base(card_id)
                        base_code_counts[base_code] = base_code_counts.get(base_code, 0) + 1
            
            # Update copy_counts: map each token_id to the count of its base code
            copy_counts.clear()
            for token_id in generated:
                if token_id not in special_ids:
                    card_id = index_to_card.get(token_id, "")
                    if card_id:
                        base_code = _normalize_card_id_to_base(card_id)
                        copy_counts[token_id] = base_code_counts.get(base_code, 0)
        
        # Build model inputs: [prompt_tensor, decoder_input, ...card_features]
        model_inputs = [prompt_tensor, decoder_input] + card_feature_inputs
        outputs = model(model_inputs, training=False)
        logits = _extract_main_logits(outputs)
        # Extract freq_hist for biasing (encourages realistic card counts)
        freq_hist = _extract_freq_hist(outputs)
        # The logits shape should be [batch, sequence_length, vocab_size]
        # We want the prediction for the next token after the current sequence
        logit_position = len(generated) - 1
        next_token_logits = logits[0, logit_position]

        next_token_logits = _apply_penalty(next_token_logits, set_penalty)
        next_token_logits = _apply_penalty(next_token_logits, prompt_bias)
        
        # Apply duplicate encouragement bias to reduce 1x cards
        # BUT: Skip this for the first token (leader generation) - only apply to main deck cards
        if len(generated) >= 2:  # Only apply after leader has been generated
            # First try freq_hist bias if available and informative
            if freq_hist is not None:
                next_token_logits = _apply_freq_hist_bias(
                    next_token_logits,
                    freq_hist,
                    copy_counts,
                    special_ids,
                    bias_strength=5.0,  # Moderate bias (reduced from 20.0 - was too strong and breaking generation)
                )
            
            # Always apply duplicate encouragement bias (doesn't rely on freq_hist)
            # This encourages cards that have been generated 1-3 times to appear again
            # AND penalizes unseen cards to force duplicates
            # Dramatically increased bias strength - model still generating all 1x cards despite previous increases
            # Increase unseen_penalty based on how many cards we've generated
            actual_card_count = sum(1 for tid in generated if tid not in special_ids)
            dynamic_unseen_penalty = 20.0
            if actual_card_count >= 40:
                dynamic_unseen_penalty = 60.0  # Much stronger penalty when we have many cards
            elif actual_card_count >= 30:
                dynamic_unseen_penalty = 40.0
            elif actual_card_count >= 20:
                dynamic_unseen_penalty = 30.0
            
            next_token_logits = _apply_duplicate_encouragement_bias(
                next_token_logits,
                copy_counts,
                special_ids,
                index_to_card=index_to_card,  # NEW: Pass index_to_card to normalize variants
                bias_strength=100.0,  # Increased from 50.0 - need even stronger boost to overcome model's overconfidence
                unseen_penalty=dynamic_unseen_penalty,  # Dynamic penalty based on card count
            )

        if repository and len(generated) >= 2:
            leader_token_id = generated[1]
            if leader_record is None:
                leader_record = _resolve_record_for_token(
                    leader_token_id, index_to_card, repository
                )
            if leader_token_id not in color_penalty_cache:
                allowed_colors = (
                    _split_colors(leader_record.color)
                    if leader_record
                    else set()
                )
                color_penalty_cache[leader_token_id] = _build_color_penalty(
                    allowed_colors,
                    card_to_index,
                    index_to_card,
                    repository,
                    special_ids,
                )
            color_penalty = color_penalty_cache.get(leader_token_id)
            cost_restriction_penalty = cost_restriction_penalty_cache.get(leader_token_id)
            if type_targets is None:
                leader_prior = _lookup_leader_prior(
                    leader_token_id, index_to_card, tournament_priors
                )
                type_targets = _compute_type_targets(
                    leader_prior, deck_config.main_deck_size
                )
        else:
            color_penalty = None
            cost_restriction_penalty = None

        if len(generated) >= 2:
            next_token_logits = _apply_penalty(next_token_logits, leader_penalty)
            next_token_logits = _apply_penalty(next_token_logits, color_penalty)
            next_token_logits = _apply_penalty(next_token_logits, cost_restriction_penalty)
            next_token_logits = _apply_copy_limit_penalty(
                next_token_logits, copy_counts, max_copies=4, special_ids=special_ids
            )
            next_token_logits = _apply_repetition_bonus(
                next_token_logits, copy_counts, special_ids=special_ids, bonus=0.5
            )
            next_token_logits = _apply_type_prior_penalty(
                next_token_logits,
                token_types,
                type_counts,
                type_targets,
                special_ids,
            )
        
        # Apply EOS penalty to discourage early termination
        # Store original logits for debugging before applying penalty
        original_logits = next_token_logits
        # Apply EOS penalty more aggressively - need to prevent early termination
        # Count actual cards (excluding special tokens)
        actual_card_count = sum(1 for tid in generated if tid not in special_ids)
        if eos_penalty > 0.0 and actual_card_count < deck_config.main_deck_size:
            # Stronger penalty when we have fewer cards
            penalty_strength = eos_penalty
            if actual_card_count < 20:
                penalty_strength = eos_penalty * 2.0  # Double penalty if less than 20 cards
            elif actual_card_count < 30:
                penalty_strength = eos_penalty * 1.5  # 1.5x penalty if less than 30 cards
            
            eos_penalty_tensor = tf.zeros_like(next_token_logits)
            eos_penalty_tensor = tf.tensor_scatter_nd_update(
                eos_penalty_tensor,
                [[end_id]],
                [-penalty_strength]
            )
            next_token_logits = next_token_logits + eos_penalty_tensor
            print(f"DEBUG: Applied EOS penalty of -{penalty_strength} (actual cards: {actual_card_count})", file=sys.stderr)
        
        # Get top 5 predictions for debugging - show BOTH original and biased logits
        debug_top_k = 5
        debug_top_k_values_orig, debug_top_k_indices_orig = tf.nn.top_k(original_logits, k=min(debug_top_k, len(original_logits)))
        debug_top_k_values_biased, debug_top_k_indices_biased = tf.nn.top_k(next_token_logits, k=min(debug_top_k, len(next_token_logits)))
        
        top_predictions_orig = [
            (int(idx), float(val), index_to_card.get(int(idx), f"<UNK:{int(idx)}>"))
            for idx, val in zip(debug_top_k_indices_orig.numpy(), debug_top_k_values_orig.numpy())
        ]
        top_predictions_biased = [
            (int(idx), float(val), index_to_card.get(int(idx), f"<UNK:{int(idx)}>"))
            for idx, val in zip(debug_top_k_indices_biased.numpy(), debug_top_k_values_biased.numpy())
        ]
        
        actual_card_count = sum(1 for tid in generated if tid not in special_ids)
        print(f"DEBUG: Step {step}, position {logit_position}, current sequence length {len(generated)}, actual cards: {actual_card_count}", file=sys.stderr)
        print(f"DEBUG: Top {debug_top_k} predictions (ORIGINAL): {top_predictions_orig}", file=sys.stderr)
        print(f"DEBUG: Top {debug_top_k} predictions (AFTER BIASING): {top_predictions_biased}", file=sys.stderr)
        
        # Show copy counts for debugging
        if len(generated) >= 2:
            unique_base_codes = {}
            for tid in generated:
                if tid not in special_ids:
                    cid = index_to_card.get(tid, "")
                    if cid:
                        base = _normalize_card_id_to_base(cid)
                        unique_base_codes[base] = unique_base_codes.get(base, 0) + 1
            duplicates = {base: count for base, count in unique_base_codes.items() if count > 1}
            if duplicates:
                print(f"DEBUG: Cards with duplicates: {duplicates}", file=sys.stderr)
            else:
                print(f"DEBUG: No duplicates yet (all {len(unique_base_codes)} cards are 1x)", file=sys.stderr)
        
        top_predictions = top_predictions_biased  # Use biased predictions for fallback logic
        
        # Apply temperature and top-k filtering
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        if top_k is not None and top_k > 0:
            top_k_values, top_k_indices = tf.nn.top_k(next_token_logits, k=min(top_k, len(next_token_logits)))
            # Create a mask: set all non-top-k logits to very negative
            mask = tf.ones_like(next_token_logits) * (-1e9)
            updates = top_k_values
            indices = tf.expand_dims(top_k_indices, 1)
            next_token_logits = tf.tensor_scatter_nd_update(mask, indices, updates)
        
        # Sample from the distribution
        # Use argmax (greedy) when temperature is extremely low (<= 0.05) for deterministic selection
        # For temperature 0.1, use sampling to prevent model collapse while still favoring high logits
        if temperature <= 0.05:
            # Greedy decoding for extremely low temperature (fully deterministic)
            next_token = int(tf.argmax(next_token_logits))
        elif temperature != 1.0 or top_k is not None:
            # Use sampling for higher temperatures
            probs = tf.nn.softmax(next_token_logits)
            next_token = int(tf.random.categorical(tf.expand_dims(probs, 0), 1)[0, 0])
        else:
            # Greedy decoding (temperature == 1.0 and no top_k)
            next_token = int(tf.argmax(next_token_logits))
        
        token_str = index_to_card.get(next_token, f"<UNK:{next_token}>")
        print(f"DEBUG: Step {step}: generated token {next_token} -> {token_str}", file=sys.stderr)
        generated.append(next_token)
        if next_token not in special_ids:
            # Debug: show what base code this card maps to
            base_code = _normalize_card_id_to_base(token_str)
            print(f"DEBUG: Card {token_str} normalized to base code: {base_code}", file=sys.stderr)
            if next_token < len(token_types):
                category = token_types[next_token]
                if category in TYPE_BUCKETS:
                    type_counts[category] += 1
        
        # Only allow early termination if we have enough cards
        actual_card_count = sum(1 for tid in generated if tid not in special_ids)
        if next_token == end_id:
            min_cards_required = max(10, deck_config.main_deck_size - 10)  # Require at least main_deck_size - 10 cards
            if actual_card_count < min_cards_required:
                print(f"DEBUG: Ignoring early EOS at step {step} (too few cards: {actual_card_count} < {min_cards_required})", file=sys.stderr)
                # Replace EOS with a random card from top predictions (excluding EOS and special tokens)
                valid_predictions = [idx for idx, _, _ in top_predictions if idx != end_id and idx not in special_ids]
                if valid_predictions:
                    next_token = valid_predictions[0]  # Use the top non-EOS prediction
                    generated[-1] = next_token
                    token_str = index_to_card.get(next_token, f"<UNK:{next_token}>")
                    print(f"DEBUG: Replaced EOS with {token_str} to continue generation", file=sys.stderr)
                else:
                    print(f"DEBUG: Early termination at step {step} (no valid predictions)", file=sys.stderr)
                    break
            else:
                print(f"DEBUG: Early termination at step {step} (enough cards: {actual_card_count})", file=sys.stderr)
                break

    # Debug: show what tokens the generated sequence maps to before decoding
    token_names = [index_to_card.get(int(idx), f"<UNK:{idx}>") for idx in generated[:10]]
    print(f"DEBUG: Generated sequence tokens (first 10): {token_names}", file=sys.stderr)
    decoded = decode_card_sequence(generated, index_to_card=index_to_card, deck_config=deck_config)
    print(f"DEBUG: Decoded sequence: {len(decoded)} tokens -> {decoded[:10] if decoded else '[]'}", file=sys.stderr)
    return decoded


def beam_search_generate(
    prompt: str,
    model: tf.keras.Model,
    prompt_vectorizer,
    card_to_index: Dict[str, int],
    index_to_card: Dict[int, str],
    deck_config: DeckConfig = DeckConfig(),
    beam_width: int = 5,
    length_penalty: float = 0.7,
    initial_sequence: Optional[List[int]] = None,
    card_features: Optional[Dict[str, tf.Tensor]] = None,
    use_card_features: bool = True,
    repository: Optional[CardRepository] = None,
    tournament_priors: Optional[Dict[str, Dict[str, object]]] = None,
    return_top_n: int = 1,
):
    prompt_tensor = _prepare_prompt_tensor(prompt, prompt_vectorizer)
    start_id = card_to_index[deck_config.start_token]
    end_id = card_to_index[deck_config.end_token]
    pad_id = card_to_index[deck_config.pad_token]
    max_length = deck_config.max_total_cards + 2

    # Prepare card features if available and model supports them
    base_model_inputs = [prompt_tensor]
    if card_features and use_card_features:
        # Card features are constant tensors, prepare them once
        cost_feat = tf.expand_dims(card_features["cost"], 0) if len(card_features["cost"].shape) == 1 else card_features["cost"]
        power_feat = tf.expand_dims(card_features["power"], 0) if len(card_features["power"].shape) == 1 else card_features["power"]
        color_feat = tf.expand_dims(card_features["color"], 0) if len(card_features["color"].shape) == 1 else card_features["color"]
        type_feat = tf.expand_dims(card_features["type"], 0) if len(card_features["type"].shape) == 1 else card_features["type"]
        ability_feat = tf.expand_dims(card_features["ability_embedding"], 0) if len(card_features["ability_embedding"].shape) == 2 else card_features["ability_embedding"]
        has_ability_feat = tf.expand_dims(card_features["has_ability"], 0) if len(card_features["has_ability"].shape) == 1 else card_features["has_ability"]
        base_model_inputs.extend([cost_feat, power_feat, color_feat, type_feat, ability_feat, has_ability_feat])

    # If initial_sequence is provided (e.g., [start_id, leader_id]), use it; otherwise start with just [start_id]
    initial_seq = initial_sequence if initial_sequence is not None else [start_id]
    beams: List[Tuple[List[int], float]] = [(list(initial_seq), 0.0)]

    special_ids: Set[int] = {start_id, end_id, pad_id}
    set_filters = _extract_set_filters(prompt)
    set_penalty = _build_set_penalty(
        set_filters,
        card_to_index,
        index_to_card,
        repository,
        special_ids,
    )
    leader_penalty = _build_leader_penalty(
        card_features,
        card_to_index,
        index_to_card,
        repository,
        special_ids,
    )
    color_penalty_cache: Dict[int, Optional[tf.Tensor]] = {}
    cost_restriction_penalty_cache: Dict[int, Optional[tf.Tensor]] = {}
    prompt_bias = _build_prompt_bias_vector(
        prompt,
        card_to_index,
        index_to_card,
        repository,
        special_ids,
    )
    token_types = _build_token_type_lookup(
        card_features,
        card_to_index,
        index_to_card,
        repository,
    )
    type_target_cache: Dict[int, Optional[Dict[str, int]]] = {}

    # Start generating from where we left off
    start_step = len(initial_seq)
    for step in range(start_step, max_length):
        candidates: List[Tuple[List[int], float]] = []
        for seq, score in beams:
            if seq[-1] == end_id:
                candidates.append((seq, score))
                continue
            decoder_input = _prepare_decoder_input(seq, pad_id, max_length)
            # Build model inputs with decoder_input
            model_inputs = base_model_inputs.copy()
            model_inputs.insert(1, decoder_input)  # Insert decoder_input after prompt_tensor
            outputs = model(model_inputs, training=False)
            logits = _extract_main_logits(outputs)
            # Extract freq_hist for biasing (encourages realistic card counts)
            freq_hist = _extract_freq_hist(outputs)
            step_logits = logits[0, len(seq) - 1]
            step_logits = _apply_penalty(step_logits, set_penalty)
            step_logits = _apply_penalty(step_logits, prompt_bias)

            if len(seq) >= 2:
                step_logits = _apply_penalty(step_logits, leader_penalty)
                if repository:
                    leader_token_id = seq[1]
                    if leader_token_id not in color_penalty_cache:
                        leader_record = _resolve_record_for_token(
                            leader_token_id, index_to_card, repository
                        )
                        allowed_colors = (
                            _split_colors(leader_record.color)
                            if leader_record
                            else set()
                        )
                        color_penalty_cache[leader_token_id] = _build_color_penalty(
                            allowed_colors,
                            card_to_index,
                            index_to_card,
                            repository,
                            special_ids,
                        )
                        # Also build cost restriction penalty
                        if leader_token_id not in cost_restriction_penalty_cache:
                            cost_restriction_penalty_cache[leader_token_id] = _build_cost_restriction_penalty(
                                leader_record,
                                card_features,
                                card_to_index,
                                index_to_card,
                                repository,
                                special_ids,
                            )
                        color_penalty = color_penalty_cache.get(leader_token_id)
                        cost_restriction_penalty = cost_restriction_penalty_cache.get(leader_token_id)
                        step_logits = _apply_penalty(step_logits, color_penalty)
                        step_logits = _apply_penalty(step_logits, cost_restriction_penalty)

                # Build copy_counts using normalized card IDs (base codes without variants)
                # This ensures variant cards (OP01-017_p1, OP01-017_r1) are counted as the same card
                base_code_counts: Dict[str, int] = {}
                for token_id in seq:
                    if token_id not in special_ids:
                        card_id = index_to_card.get(token_id, "")
                        if card_id:
                            base_code = _normalize_card_id_to_base(card_id)
                            base_code_counts[base_code] = base_code_counts.get(base_code, 0) + 1
                
                # Convert back to token_id-based Counter for compatibility
                copy_counts = Counter()
                for token_id in seq:
                    if token_id not in special_ids:
                        card_id = index_to_card.get(token_id, "")
                        if card_id:
                            base_code = _normalize_card_id_to_base(card_id)
                            copy_counts[token_id] = base_code_counts.get(base_code, 0)
                step_logits = _apply_copy_limit_penalty(
                    step_logits, copy_counts, max_copies=4, special_ids=special_ids
                )
                # Apply duplicate encouragement bias to reduce 1x cards
                if len(seq) >= 2:  # Only apply after leader has been generated
                    # First try freq_hist bias if available and informative
                    if freq_hist is not None:
                        step_logits = _apply_freq_hist_bias(
                            step_logits,
                            freq_hist,
                            copy_counts,
                            special_ids,
                            bias_strength=5.0,  # Moderate bias (reduced from 20.0 - was too strong and breaking generation)
                        )
                    
                    # Always apply duplicate encouragement bias (doesn't rely on freq_hist)
                    # AND penalizes unseen cards to force duplicates
                    # Dramatically increased bias strength - model still generating all 1x cards despite previous increases
                    step_logits = _apply_duplicate_encouragement_bias(
                        step_logits,
                        copy_counts,
                        special_ids,
                        index_to_card=index_to_card,  # NEW: Pass index_to_card to normalize variants
                        bias_strength=50.0,  # Increased from 20.0 - need much stronger bias to overcome model's diversity preference
                        unseen_penalty=20.0,  # NEW: Penalize unseen cards to force duplicates
                    )
                if len(seq) >= 2:
                    leader_token_id = seq[1]
                    if leader_token_id not in type_target_cache:
                        leader_prior = _lookup_leader_prior(
                            leader_token_id, index_to_card, tournament_priors
                        )
                        type_target_cache[leader_token_id] = _compute_type_targets(
                            leader_prior, deck_config.main_deck_size
                        )
                    type_targets = type_target_cache.get(leader_token_id)
                    type_counts = Counter(
                        token_types[token_id]
                        for token_id in seq
                        if token_id not in special_ids and token_id < len(token_types)
                    )
                    step_logits = _apply_type_prior_penalty(
                        step_logits,
                        token_types,
                        type_counts,
                        type_targets,
                        special_ids,
                    )

            log_probs = tf.nn.log_softmax(step_logits)
            topk = tf.math.top_k(log_probs, k=beam_width)
            for token_id, token_log_prob in zip(topk.indices.numpy(), topk.values.numpy()):
                new_seq = list(seq) + [int(token_id)]
                new_score = score + float(token_log_prob)
                candidates.append((new_seq, new_score))

        def _normalize(candidate: Tuple[List[int], float]) -> float:
            seq, score = candidate
            length = len(seq)
            return score / (length ** length_penalty)

        beams = sorted(candidates, key=_normalize, reverse=True)[:beam_width]
        if all(seq[-1] == end_id for seq, _ in beams):
            break

    sorted_beams = sorted(beams, key=lambda item: item[1], reverse=True)
    top_candidates = sorted_beams[: max(1, return_top_n)]

    decoded_sequences: List[List[str]] = []
    import sys

    for index, (sequence_tokens, score) in enumerate(top_candidates):
        if index == 0:
            print(
                f"DEBUG: Best beam sequence length: {len(sequence_tokens)} (score={score:.2f})",
                file=sys.stderr,
            )
            print(
                f"DEBUG: Best beam sequence (first 20): {sequence_tokens[:20]}",
                file=sys.stderr,
            )
            token_names = [
                index_to_card.get(int(idx), f"<UNK:{idx}>")
                for idx in sequence_tokens[:10]
            ]
            print(f"DEBUG: Best beam sequence tokens: {token_names}", file=sys.stderr)

        decoded = decode_card_sequence(
            sequence_tokens,
            index_to_card=index_to_card,
            deck_config=deck_config,
        )
        decoded_sequences.append(decoded)
        if index == 0:
            print(
                f"DEBUG: Decoded best sequence: {len(decoded)} tokens -> {decoded[:10] if decoded else '[]'}",
                file=sys.stderr,
            )

    if return_top_n == 1:
        return decoded_sequences[0] if decoded_sequences else []
    return decoded_sequences

