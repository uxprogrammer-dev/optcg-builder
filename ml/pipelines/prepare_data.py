"""
Command-line workflow that generates the synthetic dataset and (optionally)
prepares TensorFlow-ready vocabularies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..config import DeckConfig
from ..data import CardRepository, PromptDeckExample
from ..datasets import (
    generate_synthetic_examples,
    load_tournament_examples,
    write_examples_jsonl,
)


def _write_split_files(examples: List[PromptDeckExample], output_dir: Path) -> Dict[str, Path]:
    per_split: Dict[str, List[PromptDeckExample]] = {}
    for example in examples:
        key = example.split or "train"
        per_split.setdefault(key, []).append(example)

    paths: Dict[str, Path] = {}
    for split_name, split_examples in per_split.items():
        path = output_dir / f"synthetic_prompt_deck_{split_name}.jsonl"
        write_examples_jsonl(split_examples, path)
        paths[split_name] = path
    return paths


def _maybe_prepare_tf_assets(
    examples: List[PromptDeckExample],
    output_dir: Path,
    deck_config: DeckConfig,
) -> None:
    try:
        from ..datasets.tfdata import build_card_vocabulary, create_text_vectorizer
    except ModuleNotFoundError:
        print("TensorFlow not available; skipping TF asset export.")
        return

    prompts = [example.prompt for example in examples]
    vectorizer = create_text_vectorizer(prompts)
    card_to_index, index_to_card = build_card_vocabulary(examples, deck_config=deck_config)

    vocab_dir = output_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    prompt_vocab_path = vocab_dir / "prompt_vocabulary.txt"
    with prompt_vocab_path.open("w", encoding="utf-8") as fp:
        for token in vectorizer.get_vocabulary():
            fp.write(token + "\n")

    card_vocab_path = vocab_dir / "card_vocabulary.json"
    with card_vocab_path.open("w", encoding="utf-8") as fp:
        json.dump({"card_to_index": card_to_index, "index_to_card": index_to_card}, fp, indent=2)

    print(f"Saved prompt vocabulary to {prompt_vocab_path}")
    print(f"Saved card vocabulary to {card_vocab_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare synthetic prompt→deck dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to card data root.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ml/artifacts"), help="Directory for generated assets."
    )
    parser.add_argument("--language", type=str, default="en", help="Language code to load.")
    parser.add_argument("--samples-per-leader", type=int, default=3, help="Synthetic decks per leader.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument(
        "--include-tournament-decks",
        dest="include_tournament_decks",
        action="store_true",
        help="Include tournament decks when generating the dataset (default).",
    )
    parser.add_argument(
        "--no-tournament-decks",
        dest="include_tournament_decks",
        action="store_false",
        help="Disable tournament deck ingestion.",
    )
    parser.add_argument(
        "--tournament-dir",
        type=Path,
        default=None,
        help="Optional override for the directory containing tournament JSON files.",
    )
    parser.add_argument(
        "--export-tf-assets",
        action="store_true",
        help="Export vocabularies for TensorFlow pipelines if TensorFlow is installed.",
    )
    parser.set_defaults(include_tournament_decks=True)
    args = parser.parse_args()

    repository = CardRepository(data_root=args.data_root, language=args.language)
    examples = generate_synthetic_examples(
        repository=repository,
        samples_per_leader=args.samples_per_leader,
        seed=args.seed,
    )

    if args.include_tournament_decks:
        tournament_examples = load_tournament_examples(
            data_root=args.data_root,
            tournaments_dir=args.tournament_dir,
            language=args.language,
            repository=repository,
        )
        if tournament_examples:
            print(f"Loaded {len(tournament_examples)} tournament decks.")
            examples.extend(tournament_examples)
        else:
            print("No tournament decks were loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "synthetic_prompt_deck.jsonl"
    write_examples_jsonl(examples, dataset_path)
    print(f"Wrote combined dataset to {dataset_path}")

    split_paths = _write_split_files(examples, args.output_dir)
    for split_name, path in split_paths.items():
        print(f"- {split_name}: {path}")

    if args.export_tf_assets:
        _maybe_prepare_tf_assets(examples, args.output_dir, deck_config=DeckConfig())


if __name__ == "__main__":
    main()

