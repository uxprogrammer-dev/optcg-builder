"""
Generate descriptive statistics for the card data to inform feature design.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .card_schema import CardRecord
from .loader import CardRepository


@dataclass
class CardinalityStats:
    total_unique_cards: int
    unique_leaders: int
    colors: Dict[str, int]
    types: Dict[str, int]
    attributes: Dict[str, int]
    families: Dict[str, int]


@dataclass
class NumericStats:
    cost_histogram: Dict[str, int]
    power_histogram: Dict[str, int]


class CardDataProfile:
    """
    Create aggregations that describe the card corpus.
    """

    def __init__(self, repository: CardRepository) -> None:
        self.repository = repository
        self._cardinality: Optional[CardinalityStats] = None
        self._numeric: Optional[NumericStats] = None
        self._set_breakdown: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------ #
    def compute(self) -> None:
        cards = list(self.repository.cards.values())
        self._cardinality = self._compute_cardinality(cards)
        self._numeric = self._compute_numeric(cards)
        self._set_breakdown = self._compute_set_breakdown(cards)

    @property
    def cardinality(self) -> CardinalityStats:
        if self._cardinality is None:
            self.compute()
        assert self._cardinality
        return self._cardinality

    @property
    def numeric(self) -> NumericStats:
        if self._numeric is None:
            self.compute()
        assert self._numeric
        return self._numeric

    @property
    def set_breakdown(self) -> Mapping[str, int]:
        if self._set_breakdown is None:
            self.compute()
        assert self._set_breakdown
        return self._set_breakdown

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, object]:
        return {
            "cardinality": {
                "total_unique_cards": self.cardinality.total_unique_cards,
                "unique_leaders": self.cardinality.unique_leaders,
                "colors": self.cardinality.colors,
                "types": self.cardinality.types,
                "attributes": self.cardinality.attributes,
                "families": {
                    k: v
                    for k, v in sorted(
                        self.cardinality.families.items(), key=lambda item: item[1], reverse=True
                    )
                },
            },
            "numeric": {
                "cost_histogram": self.numeric.cost_histogram,
                "power_histogram": self.numeric.power_histogram,
            },
            "sets": dict(sorted(self.set_breakdown.items(), key=lambda item: item[1], reverse=True)),
        }

    def export(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, sort_keys=True)

    # ------------------------------------------------------------------ #
    def _compute_cardinality(self, cards: List[CardRecord]) -> CardinalityStats:
        color_counts = Counter()
        type_counts = Counter()
        attribute_counts = Counter()
        family_counts = Counter()

        unique_leaders = 0
        for card in cards:
            if card.color:
                color_counts[card.color] += 1
            if card.type:
                type_counts[card.type] += 1
            if card.attribute:
                attribute_counts[card.attribute] += 1
            if card.family:
                family_counts[card.family] += 1
            if card.is_leader:
                unique_leaders += 1

        return CardinalityStats(
            total_unique_cards=len(cards),
            unique_leaders=unique_leaders,
            colors=dict(color_counts),
            types=dict(type_counts),
            attributes=dict(attribute_counts),
            families=dict(family_counts),
        )

    def _compute_numeric(self, cards: Iterable[CardRecord]) -> NumericStats:
        cost_histogram = Counter()
        power_histogram = Counter()
        for card in cards:
            if card.cost is not None:
                cost_histogram[str(card.cost)] += 1
            else:
                cost_histogram["unknown"] += 1
            if card.power is not None:
                power_histogram[str(card.power)] += 1
            else:
                power_histogram["unknown"] += 1
        return NumericStats(
            cost_histogram=dict(cost_histogram),
            power_histogram=dict(power_histogram),
        )

    def _compute_set_breakdown(self, cards: Iterable[CardRecord]) -> Dict[str, int]:
        counts = defaultdict(int)
        for card in cards:
            if card.set_name:
                counts[card.set_name] += 1
            else:
                counts["unknown"] += 1
        return dict(counts)


def write_profile(
    data_root: Path,
    output_dir: Path = Path("ml/artifacts"),
    language: str = "en",
) -> Path:
    repo = CardRepository(data_root=data_root, language=language)
    profile = CardDataProfile(repo)
    profile.compute()
    output_path = output_dir / f"card_profile_{language}.json"
    profile.export(output_path)
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "data"
    target = write_profile(root)
    print(f"Card profile written to {target}")

