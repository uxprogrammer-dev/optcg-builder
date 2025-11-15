"""
Command-line entry point for intent classification inference.

Example usage:

python -m ml.inference.intent_classify \
    --prompt "Aggressive red Luffy deck with rush characters" \
    --model models/intent_run_20250101-120000/intent_classifier.keras \
    --prompt-vocab models/intent_run_20250101-120000/vocab/prompt_vocabulary.txt \
    --color-vocab models/intent_run_20250101-120000/vocab/color_vocabulary.json \
    --keyword-vocab models/intent_run_20250101-120000/vocab/keyword_vocabulary.json \
    --leader-vocab models/intent_run_20250101-120000/vocab/leader_vocabulary.json \
    --strategy-vocab models/intent_run_20250101-120000/vocab/strategy_vocabulary.json \
    --leader-ids models/intent_run_20250101-120000/vocab/leader_ids.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf

from ..config import PromptConfig
from ..datasets.tfdata import create_text_vectorizer_from_vocab
from ..models.intent_classifier import IntentClassifierConfig
from ..models.transformer import TransformerEncoderLayer

# Try to import register_keras_serializable
try:
    from tensorflow.keras.saving import register_keras_serializable
except (ImportError, AttributeError):
    try:
        from keras.saving import register_keras_serializable
    except (ImportError, AttributeError):
        try:
            from tensorflow.keras.utils import register_keras_serializable
        except (ImportError, AttributeError):
            def register_keras_serializable(package=None, name=None):
                def decorator(cls):
                    return cls
                return decorator


def _load_prompt_vocab(path: Path) -> List[str]:
    """Load prompt vocabulary from text file."""
    with path.open("r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def _load_json_vocab(path: Path) -> Dict:
    """Load vocabulary from JSON file."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _normalize_text(text: str) -> str:
    """Normalize text for tokenization."""
    return text.lower().replace(".", " ").replace("-", " ").strip()


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    normalized = _normalize_text(text)
    tokens = [t.strip() for t in normalized.split() if t.strip()]
    return tokens


def _extract_card_ids(prompt: str) -> List[str]:
    """Extract card IDs from prompt using regex."""
    pattern = r"(?:OP|ST|EB|P|PRB)\d{2}-\d{3}(?:_[A-Z0-9]+)?"
    matches = re.findall(pattern, prompt, re.IGNORECASE)
    return [m.upper() for m in matches]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify intent from prompt")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt to classify")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained intent classifier model")
    parser.add_argument("--prompt-vocab", type=Path, required=True, help="Path to prompt vocabulary file")
    parser.add_argument("--color-vocab", type=Path, required=True, help="Path to color vocabulary JSON")
    parser.add_argument("--keyword-vocab", type=Path, required=True, help="Path to keyword vocabulary JSON")
    parser.add_argument("--leader-vocab", type=Path, required=True, help="Path to leader vocabulary JSON")
    parser.add_argument("--strategy-vocab", type=Path, required=True, help="Path to strategy vocabulary JSON")
    parser.add_argument("--leader-ids", type=Path, required=True, help="Path to leader IDs JSON list")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification (default: 0.5)",
    )
    parser.add_argument(
        "--top-k-leaders",
        type=int,
        default=10,
        help="Number of top leaders to return (default: 10)",
    )
    parser.add_argument(
        "--top-k-keywords",
        type=int,
        default=20,
        help="Number of top keywords to return (default: 20)",
    )

    args = parser.parse_args()

    # Load vocabularies
    prompt_vocab = _load_prompt_vocab(args.prompt_vocab)
    color_vocab = _load_json_vocab(args.color_vocab)
    keyword_vocab = _load_json_vocab(args.keyword_vocab)
    leader_vocab = _load_json_vocab(args.leader_vocab)
    strategy_vocab = _load_json_vocab(args.strategy_vocab)

    with args.leader_ids.open("r") as f:
        leader_ids_list = json.load(f)

    # Create prompt vectorizer
    prompt_config = PromptConfig()
    vectorizer = create_text_vectorizer_from_vocab(prompt_vocab, prompt_config=prompt_config)

    # Load model
    # Import custom layers so they can be deserialized
    from ..models.transformer import PositionalEncodingLayer
    import ml.models.intent_classifier as intent_model_module
    
    # Enable unsafe deserialization to allow loading Lambda layers
    import keras
    keras.config.enable_unsafe_deserialization()
    
    # Define the padding mask function
    def create_padding_mask_fn(tokens):
        return tf.cast(tf.not_equal(tokens, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    
    # Monkey-patch Lambda.compute_output_shape to handle our specific case
    # This is a workaround for Lambda layers saved without output_shape
    original_compute_output_shape = tf.keras.layers.Lambda.compute_output_shape
    
    def patched_compute_output_shape(self, input_shape):
        # Check if this is our padding mask Lambda by checking the function name or layer name
        if hasattr(self, 'function') and (
            (hasattr(self.function, '__name__') and 'create_padding_mask' in str(self.function.__name__)) or
            (hasattr(self, 'name') and self.name == 'create_padding_mask')
        ):
            # Input: (batch, seq_len) -> Output: (batch, 1, 1, seq_len)
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
                return (input_shape[0], 1, 1, input_shape[1])
            return (None, 1, 1, None)
        # Fall back to original implementation
        try:
            return original_compute_output_shape(self, input_shape)
        except NotImplementedError:
            # If original fails, try to infer from function
            # For padding mask: input (batch, seq) -> output (batch, 1, 1, seq)
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
                return (input_shape[0], 1, 1, input_shape[1])
            return (None, 1, 1, None)
    
    # Temporarily patch the method
    tf.keras.layers.Lambda.compute_output_shape = patched_compute_output_shape

    # Ensure the intent classifier module has access to TensorFlow when Lambda layers run
    intent_model_module.tf = tf  # type: ignore[attr-defined]

    custom_objects = {
        "TransformerEncoderLayer": TransformerEncoderLayer,
        "PositionalEncodingLayer": PositionalEncodingLayer,
        "create_padding_mask": create_padding_mask_fn,
    }

    try:
        model = tf.keras.models.load_model(
            str(args.model),
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False,
        )
    finally:
        # Restore original method
        tf.keras.layers.Lambda.compute_output_shape = original_compute_output_shape

    # Ensure lambda layer has access to TensorFlow (its globals can be missing after deserialization)
    for layer in model.layers:
        if layer.__class__.__name__ == "Lambda" and layer.name == "create_padding_mask":
            if hasattr(layer, "function") and layer.function is not None:
                layer.function.__globals__["tf"] = tf

    # Vectorize prompt
    prompt_tensor = vectorizer(tf.constant([args.prompt]))

    # Run inference
    predictions = model(prompt_tensor, training=False)

    # Extract predictions
    color_logits = predictions["colors"][0].numpy()
    keyword_logits = predictions["keywords"][0].numpy()
    leader_logits = predictions["leaders"][0].numpy()
    strategy_logits = predictions["strategies"][0].numpy()

    # Post-process colors
    color_to_index = color_vocab["color_to_index"]
    index_to_color = color_vocab["index_to_color"]
    colors = []
    color_hints = []
    for i, score in enumerate(color_logits):
        if score >= args.threshold:
            color_name = index_to_color.get(str(i), "")
            if color_name:
                colors.append(color_name)
                color_hints.append({"color": color_name, "score": float(score)})
    color_hints.sort(key=lambda x: x["score"], reverse=True)

    # Post-process keywords
    keyword_to_index = keyword_vocab["keyword_to_index"]
    index_to_keyword = keyword_vocab["index_to_keyword"]
    keyword_scores = [(i, score) for i, score in enumerate(keyword_logits) if score >= args.threshold]
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    keywords = [index_to_keyword.get(str(i), "") for i, _ in keyword_scores[: args.top_k_keywords]]
    keywords = [k for k in keywords if k]  # Filter empty strings

    # Post-process leaders (rank by score)
    leader_to_index = leader_vocab["leader_to_index"]
    index_to_leader = leader_vocab["index_to_leader"]
    leader_scores = [(i, score) for i, score in enumerate(leader_logits) if score >= args.threshold]
    leader_scores.sort(key=lambda x: x[1], reverse=True)
    top_leader_indices = [i for i, _ in leader_scores[: args.top_k_leaders]]

    # Build leader matches with metadata
    leader_matches = []
    explicit_leader_ids = []
    for idx in top_leader_indices:
        leader_id = index_to_leader.get(str(idx), "")
        if leader_id:
            score = leader_logits[idx]
            # Try to get leader name from leader_ids_list (would need leader metadata)
            # For now, use leader_id as name
            leader_matches.append(
                {
                    "leaderId": leader_id,
                    "leaderName": leader_id,  # Will be filled by TypeScript service
                    "score": float(score),
                    "matchedKeywords": [],  # Will be filled by TypeScript service
                    "matchedColors": [],
                    "reasons": [f"ML model score: {score:.3f}"],
                }
            )
            if score >= 0.7:  # High confidence threshold for explicit
                explicit_leader_ids.append(leader_id)

    # Post-process strategies
    strategy_to_index = strategy_vocab["strategy_to_index"]
    index_to_strategy = strategy_vocab["index_to_strategy"]
    strategies = []
    for i, score in enumerate(strategy_logits):
        if score >= args.threshold:
            strategy_name = index_to_strategy.get(str(i), "")
            if strategy_name:
                strategies.append(strategy_name)

    # Extract card IDs from prompt (regex-based, same as current implementation)
    card_ids = _extract_card_ids(args.prompt)

    # Tokenize prompt
    tokens = _tokenize(args.prompt)
    normalized_prompt = _normalize_text(args.prompt)

    # Build output matching PromptIntentAnalysis format
    output = {
        "prompt": args.prompt,
        "normalizedPrompt": normalized_prompt,
        "tokens": tokens,
        "keywords": keywords,
        "cardIds": card_ids,
        "colors": colors,
        "colorHints": color_hints,
        "explicitLeaderIds": explicit_leader_ids,
        "leaderMatches": leader_matches,
        "unmatchedKeywords": [],  # Would need to compare against all possible keywords
    }

    # Output as JSON
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

