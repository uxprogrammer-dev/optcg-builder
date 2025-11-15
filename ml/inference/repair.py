"""
Deck repair utilities to enforce hard game constraints on generated decks.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..config import DeckConfig
from ..data import CardRecord, CardRepository

_COPY_LIMIT = 4
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


def _split_colors(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    cleaned = raw.replace("&", "/")
    return {part.strip().upper() for part in cleaned.split("/") if part.strip()}


def _allowed_colors_for_leader(
    leader_id: str, repository: Optional[CardRepository]
) -> Set[str]:
    if repository is None:
        return set()
    record = repository.by_id(leader_id)
    return _split_colors(record.color) if record else set()


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


def _score_card_for_prompt(
    record: Optional[CardRecord], prompt_lower: str
) -> float:
    if record is None:
        return 0.0
    text = _collect_card_text(record)
    score = 0.0
    for keyword, weight in _PROMPT_KEYWORD_WEIGHTS.items():
        if keyword in prompt_lower and keyword in text:
            score += weight
    return score


def _trim_to_size(
    cards: List[str],
    prompt_lower: str,
    repository: Optional[CardRepository],
    target_size: int,
) -> List[str]:
    if len(cards) <= target_size:
        return cards

    scoring_cache: Dict[str, float] = {}
    occurrence_tracker: Counter[str] = Counter()
    scored_instances: List[Tuple[float, int, str]] = []

    for index, card_id in enumerate(cards):
        if card_id not in scoring_cache:
            record = repository.by_id(card_id) if repository else None
            scoring_cache[card_id] = _score_card_for_prompt(record, prompt_lower)
        occurrence = occurrence_tracker[card_id]
        occurrence_tracker[card_id] += 1
        adjusted = scoring_cache[card_id] - 0.05 * occurrence - 0.001 * index
        scored_instances.append((adjusted, index, card_id))

    scored_instances.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    keep_indices = {
        index for _, index, _ in scored_instances[:target_size]
    }
    return [card for idx, card in enumerate(cards) if idx in keep_indices]


def _candidate_cards(
    repository: CardRepository,
    allowed_colors: Set[str],
    prompt_lower: str,
    existing_counts: Counter[str],
) -> List[Tuple[float, str]]:
    candidates: List[Tuple[float, str]] = []
    for card_id, record in repository.cards.items():
        if record.is_leader:
            continue
        if allowed_colors:
            card_colors = _split_colors(record.color)
            if card_colors and not (card_colors & allowed_colors):
                continue
        current_count = existing_counts[card_id]
        if current_count >= _COPY_LIMIT:
            continue
        score = _score_card_for_prompt(record, prompt_lower)
        # Slight penalty for already-present copies to encourage diversity.
        adjusted = score - 0.05 * current_count
        candidates.append((adjusted, card_id))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _fill_shortfall(
    cards: List[str],
    counts: Counter[str],
    leader_id: str,
    prompt_lower: str,
    target_size: int,
    repository: Optional[CardRepository],
) -> List[str]:
    if len(cards) >= target_size:
        return cards

    if repository is None:
        # Fallback: repeat existing cards within copy limits.
        idx = 0
        while len(cards) < target_size and cards:
            candidate = cards[idx % len(cards)]
            if counts[candidate] < _COPY_LIMIT:
                cards.append(candidate)
                counts[candidate] += 1
            idx += 1
        return cards

    allowed_colors = _allowed_colors_for_leader(leader_id, repository)
    candidates = _candidate_cards(
        repository, allowed_colors, prompt_lower, counts
    )
    for _, card_id in candidates:
        if len(cards) >= target_size:
            break
        if counts[card_id] >= _COPY_LIMIT:
            continue
        cards.append(card_id)
        counts[card_id] += 1

    # Final fallback if still short.
    idx = 0
    while len(cards) < target_size and cards:
        candidate = cards[idx % len(cards)]
        if counts[candidate] < _COPY_LIMIT:
            cards.append(candidate)
            counts[candidate] += 1
        idx += 1

    return cards


def repair_deck(
    leader_id: str,
    main_deck: Sequence[str],
    prompt: str,
    deck_config: DeckConfig,
    repository: Optional[CardRepository] = None,
) -> List[str]:
    """
    Enforce copy limits and deck size, adding replacements that match the prompt.
    """

    prompt_lower = prompt.lower()
    target_size = deck_config.main_deck_size
    kept_counts: Counter[str] = Counter()
    repaired: List[str] = []

    for card_id in main_deck:
        if not card_id or card_id == leader_id:
            continue
        if kept_counts[card_id] >= _COPY_LIMIT:
            continue
        repaired.append(card_id)
        kept_counts[card_id] += 1

    repaired = _trim_to_size(
        repaired,
        prompt_lower,
        repository,
        target_size,
    )

    kept_counts = Counter(repaired)
    repaired = _fill_shortfall(
        repaired,
        kept_counts,
        leader_id,
        prompt_lower,
        target_size,
        repository,
    )

    repaired = repaired[:target_size]
    if len(repaired) < target_size and repaired:
        filler = repaired[-1]
        while len(repaired) < target_size:
            repaired.append(filler)
    return repaired[:target_size]


__all__ = ["repair_deck"]

