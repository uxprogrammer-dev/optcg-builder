"""
Composite scoring utilities for generated decks.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import DeckConfig
from ..data import CardRecord, CardRepository

_PROMPT_KEYWORD_WEIGHTS = {
    "blocker": 0.8,
    "rush": 0.8,
    "double attack": 0.7,
    "trigger": 0.6,
    "counter": 0.6,
    "search": 0.6,
    "draw": 0.5,
    "k.o.": 0.5,
    "destroy": 0.5,
    "banish": 0.5,
    "discard": 0.5,
    "impel down": 0.8,
    "world government": 0.7,
    "marine": 0.7,
    "navy": 0.7,
    "bounty": 0.5,
}
_TYPE_BUCKETS = ("CHARACTER", "EVENT", "STAGE", "OTHER")
_COST_BUCKETS = 11  # 0..10+


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


def _normalize_type(card_type: Optional[str]) -> str:
    if not card_type:
        return "OTHER"
    value = card_type.upper()
    if value in _TYPE_BUCKETS:
        return value
    return "OTHER"


def _prompt_alignment_score(
    prompt_lower: str,
    records: Sequence[Optional[CardRecord]],
) -> Tuple[float, int]:
    active_keywords = [
        keyword for keyword in _PROMPT_KEYWORD_WEIGHTS if keyword in prompt_lower
    ]
    if not active_keywords:
        return 0.0, 0

    score = 0.0
    hits = set()

    for record in records:
        text = _collect_card_text(record)
        for keyword in active_keywords:
            if keyword in text:
                score += _PROMPT_KEYWORD_WEIGHTS[keyword]
                hits.add(keyword)

    return score, len(hits)


def _type_ratios(records: Sequence[Optional[CardRecord]]) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for record in records:
        if record is None or record.is_leader:
            continue
        bucket = _normalize_type(record.type)
        counts[bucket] += 1
    total = sum(counts.values()) or 1
    return {bucket: counts.get(bucket, 0) / total for bucket in _TYPE_BUCKETS}


def _cost_ratios(records: Sequence[Optional[CardRecord]]) -> List[float]:
    histogram = [0] * _COST_BUCKETS
    for record in records:
        if record is None or record.is_leader:
            continue
        cost = record.cost if record.cost is not None else 0
        bucket = cost if 0 <= cost < _COST_BUCKETS - 1 else _COST_BUCKETS - 1
        histogram[bucket] += 1
    total = sum(histogram) or 1
    return [value / total for value in histogram]


def _distribution_distance(
    actual: Mapping[str, float],
    target: Optional[Mapping[str, float]],
) -> float:
    if not target:
        return 0.0
    distance = 0.0
    for key in actual:
        distance += abs(actual.get(key, 0.0) - float(target.get(key, 0.0)))
    return distance


def _cost_distance(
    actual: Sequence[float],
    target: Optional[Sequence[float]],
) -> float:
    if not target:
        return 0.0
    distance = 0.0
    length = min(len(actual), len(target))
    for index in range(length):
        distance += abs(actual[index] - float(target[index]))
    return distance


def _legality_penalty(
    leader_id: str,
    main_deck: Sequence[str],
    deck_config: DeckConfig,
) -> float:
    penalty = 0.0
    if len(main_deck) != deck_config.main_deck_size:
        penalty += 1.0
    counts = Counter(card_id for card_id in main_deck if card_id != leader_id)
    for count in counts.values():
        if count > 4:
            penalty += (count - 4)
    return penalty


def _lookup_prior(
    leader_id: str,
    priors: Optional[Mapping[str, Mapping[str, object]]],
) -> Optional[Mapping[str, object]]:
    if not priors:
        return None
    if leader_id in priors:
        return priors[leader_id]
    base_id = leader_id.split("_")[0]
    return priors.get(base_id)


def score_sequence(
    prompt: str,
    sequence: Sequence[str],
    deck_config: DeckConfig,
    repository: Optional[CardRepository] = None,
    priors: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> float:
    if not sequence:
        return float("-inf")

    leader_id = sequence[0]
    main_deck = sequence[1 : 1 + deck_config.main_deck_size]

    records: List[Optional[CardRecord]] = []
    if repository:
        records = [repository.by_id(card_id) for card_id in main_deck]

    prompt_lower = prompt.lower()
    prompt_score, keyword_hits = _prompt_alignment_score(prompt_lower, records)

    legality = _legality_penalty(leader_id, main_deck, deck_config)

    type_ratio = _type_ratios(records)
    cost_ratio = _cost_ratios(records)

    prior = _lookup_prior(leader_id, priors)
    type_distance = _distribution_distance(
        type_ratio, prior.get("avg_type_ratio") if prior else None
    )
    cost_distance_value = _cost_distance(
        cost_ratio, prior.get("avg_cost_ratio") if prior else None
    )

    score = 0.0
    score += prompt_score * 5.0
    score += keyword_hits * 1.5
    score -= legality * 50.0
    score -= type_distance * 20.0
    score -= cost_distance_value * 10.0

    return score


def rerank_candidates(
    prompt: str,
    candidates: Sequence[Sequence[str]],
    deck_config: DeckConfig,
    repository: Optional[CardRepository] = None,
    priors: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Sequence[str]:
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates[0]

    best_sequence: Sequence[str] = candidates[0]
    best_score = float("-inf")

    for sequence in candidates:
        score = score_sequence(
            prompt=prompt,
            sequence=sequence,
            deck_config=deck_config,
            repository=repository,
            priors=priors,
        )
        if score > best_score:
            best_score = score
            best_sequence = sequence

    return best_sequence


__all__ = ["score_sequence", "rerank_candidates"]

