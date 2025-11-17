"""
Evaluation entrypoint comparing generated decks against a reference dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import tensorflow as tf

from ..config import DeckConfig
from ..data import CardRepository, DeckSchema
from ..datasets import (
    build_card_feature_matrix,
    create_ability_vectorizer,
    create_text_vectorizer_from_vocab,
    load_examples_jsonl,
)
from ..inference import beam_search_generate, greedy_generate
from ..evaluation.metrics import EvaluationReport, evaluate_predictions, sequence_to_deck
# Import custom layers so they can be deserialized when loading the model
from ..models.transformer import (
    CombinedDecoderMaskLayer,
    ExtractFeatureLayer,
    PaddingMaskLayer,
    PositionalEncodingLayer,
    PredictedSequenceFreqHistLayer,  # Phase 1: Sequence-level loss layer
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    gather_card_features,
)


def _load_card_vocab(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    card_to_index = payload["card_to_index"]
    index_to_card = {int(k): v for k, v in payload["index_to_card"].items()}
    return {"card_to_index": card_to_index, "index_to_card": index_to_card}


def _decode_strategy(name: str):
    if name == "beam":
        return beam_search_generate
    if name == "greedy":
        return greedy_generate
    raise ValueError(f"Unknown decode strategy: {name}")


def _is_trivial_sequence(tokens: List[str] | None, deck_config: DeckConfig) -> bool:
    """
    Determine whether a generated token sequence is effectively empty
    (only special tokens such as <BOS>/<EOS>/<PAD>).
    """
    if not tokens:
        return True

    special = {deck_config.start_token, deck_config.end_token, deck_config.pad_token}
    meaningful = [token for token in tokens if token not in special]
    return len(meaningful) == 0


def evaluate(
    dataset_path: Path,
    model_path: Path,
    prompt_vocab_path: Path,
    card_vocab_path: Path,
    output_path: Path,
    split: str = "test",
    decode_strategy: str = "beam",
    beam_width: int = 5,
) -> EvaluationReport:
    examples = load_examples_jsonl(dataset_path)
    subset = [example for example in examples if (example.split or "train") == split]
    if not subset:
        raise ValueError(f"No examples found for split '{split}'.")

    with prompt_vocab_path.open("r", encoding="utf-8") as fp:
        prompt_vocab = [line.strip() for line in fp if line.strip()]

    card_vocab = _load_card_vocab(card_vocab_path)
    card_to_index = card_vocab["card_to_index"]
    index_to_card = card_vocab["index_to_card"]

    prompt_vectorizer = create_text_vectorizer_from_vocab(prompt_vocab)
    
    # Enable unsafe deserialization to allow loading custom layers
    import keras
    keras.config.enable_unsafe_deserialization()
    
    # Provide custom objects explicitly to ensure they can be deserialized
    custom_objects = {
        "ExtractFeatureLayer": ExtractFeatureLayer,
        "TransformerEncoderLayer": TransformerEncoderLayer,
        "TransformerDecoderLayer": TransformerDecoderLayer,
        "PredictedSequenceFreqHistLayer": PredictedSequenceFreqHistLayer,  # Phase 1: Sequence-level loss layer
        "PositionalEncodingLayer": PositionalEncodingLayer,
        "PaddingMaskLayer": PaddingMaskLayer,
        "CombinedDecoderMaskLayer": CombinedDecoderMaskLayer,
        "gather_card_features": gather_card_features,
    }
    
    model = tf.keras.models.load_model(
        model_path, 
        compile=False, 
        custom_objects=custom_objects,
        safe_mode=False
    )
    
    # Check if model requires card features (models with card features have 8 inputs)
    model_input_count = len(model.inputs) if hasattr(model, 'inputs') and model.inputs else 0
    use_card_features_in_model = model_input_count >= 8
    
    deck_config = DeckConfig()
    repository = CardRepository(data_root=Path("data"))
    repository.load()
    
    # Build card features if model requires them
    card_features = None
    if use_card_features_in_model:
        try:
            ability_vectorizer = create_ability_vectorizer(repository, vocab_size=2000, max_length=64)
            card_features = build_card_feature_matrix(
                card_to_index,
                repository,
                ability_vectorizer=ability_vectorizer,
                max_ability_length=64,
            )
            print(f"Built card features for {len(card_to_index)} cards")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to build card features: {e}. Evaluation may fail.")
            print(f"WARNING: Could not build card features: {e}")

    predictions = []
    targets = []

    for example in subset:
        if decode_strategy == "beam":
            prediction_tokens = beam_search_generate(
                prompt=example.prompt,
                model=model,
                prompt_vectorizer=prompt_vectorizer,
                card_to_index=card_to_index,
                index_to_card=index_to_card,
                deck_config=deck_config,
                beam_width=beam_width,
                card_features=card_features,
                use_card_features=use_card_features_in_model,
                repository=repository,
            )
            # If beam search collapses to BOS/EOS, fall back to greedy decoding
            if _is_trivial_sequence(prediction_tokens, deck_config):
                prediction_tokens = greedy_generate(
                    prompt=example.prompt,
                    model=model,
                    prompt_vectorizer=prompt_vectorizer,
                    card_to_index=card_to_index,
                    index_to_card=index_to_card,
                    deck_config=deck_config,
                    card_features=card_features,
                    use_card_features=use_card_features_in_model,
                    repository=repository,
                )
        else:
            prediction_tokens = greedy_generate(
                prompt=example.prompt,
                model=model,
                prompt_vectorizer=prompt_vectorizer,
                card_to_index=card_to_index,
                index_to_card=index_to_card,
                deck_config=deck_config,
                card_features=card_features,
                use_card_features=use_card_features_in_model,
                repository=repository,
            )
        predicted_deck = sequence_to_deck(prediction_tokens, deck_config=deck_config)
        target_deck = DeckSchema(
            leader_id=example.deck.leader_id,
            main_deck=example.deck.main_deck,
            sideboard=example.deck.sideboard,
            metadata=example.deck.metadata,
        )
        predictions.append(predicted_deck)
        targets.append(target_deck)

    report = evaluate_predictions(
        predictions,
        targets,
        repository=repository,
        deck_config=deck_config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(report.__dict__, fp, indent=2)

    print(f"Evaluation complete. Report saved to {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deck generation model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("ml/artifacts/synthetic_prompt_deck.jsonl"),
        help="Path to JSONL dataset with references.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to trained Keras model.")
    parser.add_argument("--prompt-vocab", type=Path, required=True, help="Path to prompt vocabulary text file.")
    parser.add_argument("--card-vocab", type=Path, required=True, help="Path to card vocabulary JSON.")
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/eval_report.json"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--decode-strategy", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam-width", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        dataset_path=args.dataset,
        model_path=args.model,
        prompt_vocab_path=args.prompt_vocab,
        card_vocab_path=args.card_vocab,
        output_path=args.output,
        split=args.split,
        decode_strategy=args.decode_strategy,
        beam_width=args.beam_width,
    )


if __name__ == "__main__":
    main()

