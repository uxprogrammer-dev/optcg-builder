#!/usr/bin/env python
"""
Merge tournament decklists into the synthetic prompt→deck dataset.

Usage:
    python scripts/merge_tournament_decks.py \
        --synthetic ml/artifacts/synthetic_prompt_deck.jsonl \
        --tournament-dir data/tournaments \
        --output ml/artifacts/combined_prompt_deck.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import DeckConfig
from ml.data import CardRepository, DeckSchema, PromptDeckExample


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False))
            fp.write("\n")


def _split_fields(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.replace(",", "/").split("/") if part.strip()]


def _build_prompt(entry: Dict[str, object], leader_name: str, leader_id: str, colors: List[str]) -> str:
    color_text = "/".join(colors) if colors else "multicolor"
    deck_name = entry.get("deck_name") or "Unnamed deck"
    tournament = entry.get("tournament") or "Unknown tournament"
    placement = entry.get("placement") or ""
    author = entry.get("author") or "Unknown player"
    date = entry.get("date") or "Unknown date"
    country = entry.get("country") or "Unknown country"

    header = [
        f"Leader: {leader_name} ({leader_id})",
        f"Deck Name: {deck_name}",
        f"Tournament: {tournament} {placement}".strip(),
        f"Player: {author}",
        f"Date: {date} • Country: {country}",
    ]
    header_text = "\n".join(header)
    body = (
        f"Build a competitive {color_text} deck around {leader_name} that reflects the tournament strategy "
        f"demonstrated by this result."
    )
    return f"{header_text}\n\n{body}"


def _find_card(repo: CardRepository, card_id: str):
    card = repo.by_id(card_id)
    if card:
        return card
    matches = repo.by_code(card_id)
    if matches:
        return matches[0]
    return None


def load_tournament_examples(
    tournament_dir: Path,
    repo: CardRepository,
    split: str = "train",
) -> List[Dict[str, object]]:
    deck_config = DeckConfig()
    examples: List[Dict[str, object]] = []

    for json_path in sorted(tournament_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, list):
            continue

        for entry in payload:
            decklist: Dict[str, int] = entry.get("decklist") or {}
            leader_card = None
            leader_id: Optional[str] = None
            main_cards: List[str] = []

            for raw_id, count in decklist.items():
                card = _find_card(repo, raw_id)
                if not card or not isinstance(count, int):
                    continue
                if card.is_leader and leader_id is None:
                    leader_card = card
                    leader_id = card.id
                    continue
                main_cards.extend([card.id] * count)

            if not leader_id or not leader_card:
                continue

            if len(main_cards) < deck_config.main_deck_size:
                continue

            # Deterministic ordering
            main_cards = main_cards[: deck_config.main_deck_size]

            metadata = {
                "source": "tournament",
                "deck_name": entry.get("deck_name"),
                "author": entry.get("author"),
                "date": entry.get("date"),
                "country": entry.get("country"),
                "tournament": entry.get("tournament"),
                "placement": entry.get("placement"),
                "host": entry.get("host"),
                "source_file": json_path.name,
            }

            deck = DeckSchema(leader_id=leader_id, main_deck=main_cards, metadata=metadata)

            colors = _split_fields(leader_card.color)
            subtypes = _split_fields(leader_card.family)

            prompt = _build_prompt(entry, leader_card.name, leader_id, colors)

            example = PromptDeckExample(
                prompt=prompt,
                deck=deck,
                prompt_style="tournament",
                quality_tags=["tournament"],
                split=split,
                leader_ability=leader_card.ability,
                leader_subtypes=subtypes,
                leader_colors=colors,
            )
            examples.append(example.to_record())

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Append tournament decks to the synthetic prompt→deck dataset.")
    parser.add_argument("--synthetic", type=Path, required=True, help="Path to the base synthetic JSONL file.")
    parser.add_argument(
        "--tournament-dir",
        type=Path,
        required=True,
        help="Directory containing tournament JSON files (e.g., data/tournaments).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL file for the merged dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to assign to tournament examples (default: train).",
    )
    args = parser.parse_args()

    synthetic_records = _load_jsonl(args.synthetic)

    repo = CardRepository(data_root=Path("data"), language="en")
    repo.load()

    tournament_examples = load_tournament_examples(args.tournament_dir, repo, split=args.split)

    combined = synthetic_records + tournament_examples
    _write_jsonl(args.output, combined)

    print(f"Synthetic examples: {len(synthetic_records)}")
    print(f"Tournament examples: {len(tournament_examples)}")
    print(f"Total combined: {len(combined)}")
    print(f"Merged dataset written to {args.output}")


if __name__ == "__main__":
    main()

