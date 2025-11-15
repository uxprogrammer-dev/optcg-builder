"""
Compute tournament-derived priors for deck generation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

from ..data import CardRecord, CardRepository
from ..datasets.tournament import load_tournament_examples

TYPE_BUCKETS = ("CHARACTER", "EVENT", "STAGE", "OTHER")
KEYWORDS = (
    "blocker",
    "rush",
    "double attack",
    "trigger",
    "counter",
    "search",
    "draw",
    "k.o.",
    "destroy",
    "banish",
    "discard",
    "impel down",
    "world government",
    "marine",
    "navy",
    "bounty",
)
COST_BUCKETS = 11  # 0..10+


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
    card_type_upper = card_type.upper()
    if card_type_upper in TYPE_BUCKETS:
        return card_type_upper
    return "OTHER"


def compute_deck_profile(
    card_ids: Sequence[str],
    repository: CardRepository,
) -> Mapping[str, object]:
    type_counts: Counter[str] = Counter()
    cost_curve = [0] * COST_BUCKETS
    keyword_counts: Counter[str] = Counter()

    for card_id in card_ids:
        record = repository.by_id(card_id)
        if record is None or record.is_leader:
            continue

        card_type = _normalize_type(record.type)
        type_counts[card_type] += 1

        cost = record.cost if record.cost is not None else 0
        cost_bucket = cost if 0 <= cost < COST_BUCKETS - 1 else COST_BUCKETS - 1
        cost_curve[cost_bucket] += 1

        text = _collect_card_text(record)
        for keyword in KEYWORDS:
            if keyword in text:
                keyword_counts[keyword] += 1

    total_cards = sum(type_counts.values()) or 1
    cost_total = sum(cost_curve) or 1

    type_ratios = {bucket: type_counts.get(bucket, 0) / total_cards for bucket in TYPE_BUCKETS}
    cost_ratios = [value / cost_total for value in cost_curve]
    keyword_ratios = {keyword: keyword_counts.get(keyword, 0) / total_cards for keyword in KEYWORDS}

    return {
        "type_ratios": type_ratios,
        "cost_ratios": cost_ratios,
        "keyword_ratios": keyword_ratios,
    }


def aggregate_profiles(
    profiles_by_leader: Mapping[str, Sequence[Mapping[str, object]]],
    repository: CardRepository,
) -> Dict[str, Dict[str, object]]:
    aggregated: Dict[str, Dict[str, object]] = {}
    for leader_id, profiles in profiles_by_leader.items():
        if not profiles:
            continue
        count = len(profiles)
        type_accumulator: Counter[str] = Counter()
        cost_accumulator = [0.0] * COST_BUCKETS
        keyword_accumulator: Counter[str] = Counter()

        for profile in profiles:
            type_ratios = profile["type_ratios"]
            for bucket in TYPE_BUCKETS:
                type_accumulator[bucket] += type_ratios.get(bucket, 0.0)

            cost_ratios = profile["cost_ratios"]
            for index, value in enumerate(cost_ratios):
                cost_accumulator[index] += float(value)

            keyword_ratios = profile["keyword_ratios"]
            for keyword in KEYWORDS:
                keyword_accumulator[keyword] += keyword_ratios.get(keyword, 0.0)

        leader_record = repository.by_id(leader_id)
        aggregated[leader_id] = {
            "leader_name": leader_record.name if leader_record else leader_id,
            "samples": count,
            "avg_type_ratio": {
                bucket: type_accumulator.get(bucket, 0.0) / count for bucket in TYPE_BUCKETS
            },
            "avg_cost_ratio": [value / count for value in cost_accumulator],
            "avg_keyword_ratio": {
                keyword: keyword_accumulator.get(keyword, 0.0) / count for keyword in KEYWORDS
            },
        }
    return aggregated


def compute_tournament_priors(
    data_root: Path,
    language: str = "en",
) -> Dict[str, Dict[str, object]]:
    repository = CardRepository(data_root=data_root, language=language)
    repository.load()

    examples = load_tournament_examples(
        data_root=data_root,
        language=language,
        repository=repository,
    )

    profiles_by_leader: MutableMapping[str, List[Mapping[str, object]]] = defaultdict(list)
    for example in examples:
        leader_id = example.deck.leader_id
        profile = compute_deck_profile(example.deck.main_deck, repository)
        profiles_by_leader[leader_id].append(profile)

    return aggregate_profiles(profiles_by_leader, repository)


def save_priors(priors: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(priors, fp, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tournament priors for deck generation.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to data directory.")
    parser.add_argument("--language", default="en", help="Card language.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON path for priors (e.g., models/stats/tournament_priors.json).",
    )
    args = parser.parse_args()

    priors = compute_tournament_priors(data_root=args.data_root, language=args.language)
    save_priors(priors, args.output)
    print(f"Saved tournament priors for {len(priors)} leaders to {args.output}")


if __name__ == "__main__":
    main()

