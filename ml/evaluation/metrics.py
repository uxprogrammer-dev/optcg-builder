"""
Evaluation helpers for generated decks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..config import DeckConfig
from ..data import CardRepository, DeckSchema


@dataclass
class DeckComparison:
    exact_match: bool
    leader_match: bool
    main_deck_overlap: float
    legality: bool
    duplicate_violation: bool
    unknown_cards: int


@dataclass
class EvaluationReport:
    exact_match_rate: float
    leader_match_rate: float
    average_overlap: float
    legality_rate: float
    duplicate_violation_rate: float
    average_unknown_cards: float
    total_samples: int


def sequence_to_deck(tokens: Sequence[str], deck_config: DeckConfig) -> DeckSchema:
    filtered = [token for token in tokens if token and not token.startswith("<")]
    if not filtered:
        return DeckSchema(leader_id="", main_deck=[])
    leader_id = filtered[0]
    main_deck = list(filtered[1 : 1 + deck_config.main_deck_size])
    sideboard = filtered[1 + deck_config.main_deck_size :]
    return DeckSchema(
        leader_id=leader_id,
        main_deck=main_deck,
        sideboard=sideboard or None,
        metadata={"source": "prediction"},
    )


def _check_legality(
    deck: DeckSchema,
    repository: CardRepository,
    deck_config: DeckConfig,
    max_duplicates: int = 4,
) -> Dict[str, bool]:
    legality = True
    duplicate_violation = False
    unknown_cards = 0

    if not deck.leader_id or not repository.by_id(deck.leader_id):
        legality = False

    if deck.deck_size != deck_config.main_deck_size:
        legality = False

    counts: Dict[str, int] = {}
    for card_id in deck.main_deck:
        record = repository.by_id(card_id)
        if record is None:
            unknown_cards += 1
            legality = False
            continue
        counts[card_id] = counts.get(card_id, 0) + 1
        if counts[card_id] > max_duplicates:
            duplicate_violation = True
            legality = False

    return {"legality": legality, "duplicate_violation": duplicate_violation, "unknown_cards": unknown_cards}


def compare_decks(
    predicted: DeckSchema,
    target: DeckSchema,
    repository: CardRepository,
    deck_config: DeckConfig,
    max_duplicates: int = 4,
) -> DeckComparison:
    legality_info = _check_legality(
        predicted, repository, deck_config=deck_config, max_duplicates=max_duplicates
    )
    predicted_set = set(predicted.main_deck)
    target_set = set(target.main_deck)
    intersection = len(predicted_set & target_set)
    union = len(predicted_set | target_set) or 1
    overlap = intersection / union

    return DeckComparison(
        exact_match=predicted.leader_id == target.leader_id and predicted_set == target_set,
        leader_match=predicted.leader_id == target.leader_id,
        main_deck_overlap=overlap,
        legality=legality_info["legality"],
        duplicate_violation=legality_info["duplicate_violation"],
        unknown_cards=legality_info["unknown_cards"],
    )


def evaluate_predictions(
    predictions: Iterable[DeckSchema],
    targets: Iterable[DeckSchema],
    repository: CardRepository,
    deck_config: DeckConfig,
    max_duplicates: int = 4,
) -> EvaluationReport:
    comparisons: List[DeckComparison] = []
    for pred, target in zip(predictions, targets):
        comparisons.append(
            compare_decks(pred, target, repository=repository, deck_config=deck_config, max_duplicates=max_duplicates)
        )

    total = len(comparisons) or 1

    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / (len(values) or 1)

    return EvaluationReport(
        exact_match_rate=_mean(1.0 if c.exact_match else 0.0 for c in comparisons),
        leader_match_rate=_mean(1.0 if c.leader_match else 0.0 for c in comparisons),
        average_overlap=_mean(c.main_deck_overlap for c in comparisons),
        legality_rate=_mean(1.0 if c.legality else 0.0 for c in comparisons),
        duplicate_violation_rate=_mean(1.0 if c.duplicate_violation else 0.0 for c in comparisons),
        average_unknown_cards=_mean(float(c.unknown_cards) for c in comparisons),
        total_samples=len(comparisons),
    )

