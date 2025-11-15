"""
Script to generate intent classification training data from existing deck examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..data import CardRepository
from ..data.intent_schema import IntentExample
from ..datasets.intent_data import generate_intent_from_deck_examples, save_intent_examples
from ..datasets.synthetic import generate_synthetic_examples, load_examples_jsonl
from ..datasets.tournament import load_tournament_examples


def load_leader_keywords(data_root: Path) -> Dict[str, List[str]]:
    """
    Load leader keywords from leaders.json file.
    
    Returns:
        Mapping of leader_id -> list of keywords
    """
    leaders_path = data_root / "leaders.json"
    if not leaders_path.exists():
        return {}

    with leaders_path.open("r", encoding="utf-8") as f:
        leaders_data = json.load(f)

    leader_keywords_map: Dict[str, List[str]] = {}
    for leader_id, leader_info in leaders_data.items():
        # Extract keywords from leader metadata
        keywords = []
        if "keywords" in leader_info:
            keywords.extend(leader_info["keywords"])
        if "subtypes" in leader_info:
            keywords.extend([s.lower() for s in leader_info["subtypes"]])
        if "name" in leader_info:
            # Tokenize leader name
            name = leader_info["name"]
            name_tokens = name.lower().replace(".", " ").replace("-", " ").split()
            keywords.extend([t for t in name_tokens if len(t) >= 3])

        leader_keywords_map[leader_id.upper()] = list(set(keywords))

    return leader_keywords_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate intent classification training data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Data root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts"),
        help="Output directory for intent examples",
    )
    parser.add_argument(
        "--include-tournament",
        action="store_true",
        default=True,
        help="Include tournament deck examples",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        default=True,
        help="Include synthetic deck examples",
    )
    parser.add_argument(
        "--synthetic-samples-per-leader",
        type=int,
        default=3,
        help="Number of synthetic examples per leader",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=1234,
        help="Random seed for synthetic generation",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for card data",
    )

    args = parser.parse_args()

    # Load card repository
    repository = CardRepository(data_root=args.data_root, language=args.language)
    repository.load()

    # Load leader keywords
    leader_keywords_map = load_leader_keywords(args.data_root)

    # Collect deck examples
    deck_examples: List = []

    # Load tournament examples
    if args.include_tournament:
        print("Loading tournament examples...")
        tournament_examples = load_tournament_examples(
            data_root=args.data_root,
            language=args.language,
            repository=repository,
        )
        deck_examples.extend(tournament_examples)
        print(f"Loaded {len(tournament_examples)} tournament examples")

    # Generate synthetic examples
    if args.include_synthetic:
        print("Generating synthetic examples...")
        synthetic_examples = generate_synthetic_examples(
            repository=repository,
            samples_per_leader=args.synthetic_samples_per_leader,
            seed=args.synthetic_seed,
        )
        deck_examples.extend(synthetic_examples)
        print(f"Generated {len(synthetic_examples)} synthetic examples")

    # Convert to intent examples
    print("Converting to intent examples...")
    intent_examples = generate_intent_from_deck_examples(
        deck_examples,
        leader_keywords_map=leader_keywords_map if leader_keywords_map else None,
    )

    # Split examples
    train_examples = [ex for ex in intent_examples if (ex.split or "train") == "train"]
    val_examples = [ex for ex in intent_examples if ex.split == "val"]
    test_examples = [ex for ex in intent_examples if ex.split == "test"]

    # If no explicit splits, split randomly
    if not val_examples and not test_examples:
        total = len(intent_examples)
        val_size = max(1, total // 10)
        test_size = max(1, total // 20)
        train_size = total - val_size - test_size

        train_examples = intent_examples[:train_size]
        val_examples = intent_examples[train_size : train_size + val_size]
        test_examples = intent_examples[train_size + val_size :]

    # Save examples
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "intent_train.jsonl"
    val_path = args.output_dir / "intent_val.jsonl"
    test_path = args.output_dir / "intent_test.jsonl"

    save_intent_examples(train_examples, train_path)
    save_intent_examples(val_examples, val_path)
    save_intent_examples(test_examples, test_path)

    print(f"\nSaved intent examples:")
    print(f"  Train: {len(train_examples)} examples -> {train_path}")
    print(f"  Val: {len(val_examples)} examples -> {val_path}")
    print(f"  Test: {len(test_examples)} examples -> {test_path}")


if __name__ == "__main__":
    main()

