"""
Command-line entry point for generating a deck from a natural-language prompt.

Example usage:

python -m ml.inference.generate_deck \
    --prompt "Aggro red Straw Hat deck with rush mechanics" \
    --model models/run_20250101-120000/deck_transformer.keras \
    --prompt-vocab models/run_20250101-120000/vocab/prompt_vocabulary.txt \
    --card-vocab models/run_20250101-120000/vocab/card_vocabulary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import tensorflow as tf

from ..config import DeckConfig, PromptConfig
from ..data import CardRepository
from ..datasets import (
    build_card_feature_matrix,
    create_ability_vectorizer,
    create_text_vectorizer_from_vocab,
)
from ..models.transformer import (
    PositionalEncodingLayer,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    gather_card_features,
)  # Import custom layers so they can be deserialized
from .generate import beam_search_generate, greedy_generate
from .repair import repair_deck
from .scoring import rerank_candidates

TYPE_BUCKETS = ("CHARACTER", "EVENT", "STAGE", "OTHER")
COST_BUCKETS = 11


def _format_control_tokens(type_ratio: Sequence[float], cost_ratio: Sequence[float]) -> str:
    type_parts = [f"{bucket[:2]}={value:.2f}" for bucket, value in zip(TYPE_BUCKETS, type_ratio)]
    low = sum(cost_ratio[:3])
    mid = sum(cost_ratio[3:6])
    high = sum(cost_ratio[6:])
    cost_parts = [f"L={low:.2f}", f"M={mid:.2f}", f"H={high:.2f}"]
    return f"[TYPE:{','.join(type_parts)}] [COST:{','.join(cost_parts)}]"


def _aggregate_prior_tokens(
    leader_ids: Sequence[Optional[str]],
    priors: Mapping[str, Mapping[str, object]],
) -> str:
    if not priors:
        return ""

    type_accum = [0.0] * len(TYPE_BUCKETS)
    cost_accum = [0.0] * COST_BUCKETS
    count = 0

    for leader_id in leader_ids:
        if not leader_id:
            continue
        prior = priors.get(leader_id) or priors.get(leader_id.split("_")[0])
        if not prior:
            continue
        type_ratio = prior.get("avg_type_ratio")
        cost_ratio = prior.get("avg_cost_ratio")
        if not isinstance(type_ratio, Mapping) or not isinstance(cost_ratio, Sequence):
            continue
        type_vector = [float(type_ratio.get(bucket, 0.0)) for bucket in TYPE_BUCKETS]
        cost_vector = [float(value) for value in cost_ratio[:COST_BUCKETS]]
        if len(cost_vector) < COST_BUCKETS:
            cost_vector.extend([0.0] * (COST_BUCKETS - len(cost_vector)))
        type_accum = [a + b for a, b in zip(type_accum, type_vector)]
        cost_accum = [a + b for a, b in zip(cost_accum, cost_vector)]
        count += 1

    if count == 0:
        return ""

    type_avg = [value / count for value in type_accum]
    cost_avg = [value / count for value in cost_accum]
    return _format_control_tokens(type_avg, cost_avg)


def _load_prompt_vocab(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def _load_card_vocab(path: Path) -> Dict[str, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return {
        "card_to_index": {str(k): int(v) for k, v in payload["card_to_index"].items()},
        "index_to_card": {int(k): str(v) for k, v in payload["index_to_card"].items()},
    }


# Provide a compatible implementation of the decoder mask Lambda used in saved models
# so Keras can deserialize it safely.
def _create_combined_decoder_mask(padding_mask: tf.Tensor) -> tf.Tensor:
    # padding_mask shape: [batch, 1, 1, seq_len]
    seq_len = tf.shape(padding_mask)[-1]
    look_ahead = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    look_ahead = 1.0 - look_ahead
    look_ahead = tf.cast(look_ahead, tf.float32)
    look_ahead = tf.reshape(look_ahead, (1, 1, seq_len, seq_len))
    return tf.maximum(padding_mask, look_ahead)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deck using the trained transformer model.")
    parser.add_argument("--prompt", required=True, help="User prompt describing the desired deck.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the saved Keras model (.keras).")
    parser.add_argument(
        "--prompt-vocab",
        required=True,
        type=Path,
        help="Path to the prompt vocabulary text file (one token per line).",
    )
    parser.add_argument(
        "--card-vocab",
        required=True,
        type=Path,
        help="Path to the card vocabulary JSON generated during training.",
    )
    parser.add_argument("--decode-strategy", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.7)
    parser.add_argument("--leader-id", type=str, help="Optional leader identifier to reinforce via the prompt.")
    parser.add_argument("--leader-only", action="store_true", help="Return only leader_id using single-step decode.")
    parser.add_argument(
        "--first-token-allowlist",
        type=str,
        default="",
        help="Comma-separated list of allowed first tokens (card IDs) to restrict the first generated token.",
    )
    parser.add_argument(
        "--leader-temperature",
        type=float,
        default=1.5,
        help="Temperature for leader sampling (higher = more diversity). Default: 1.5",
    )
    parser.add_argument(
        "--leader-top-k",
        type=int,
        default=10,
        help="Top-k leaders to consider for sampling. Default: 10",
    )
    parser.add_argument(
        "--rerank-beams",
        type=int,
        default=1,
        help="Number of top beam candidates to rerank (beam search only).",
    )
    parser.add_argument(
        "--tournament-priors",
        type=Path,
        help="Optional JSON file with tournament priors (e.g., models/stats/tournament_priors.json).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Path to data directory containing cards/ (for card features).",
    )
    return parser


def main() -> None:
    # Set random seeds for reproducibility, but use time-based seed for diversity
    import random
    import numpy as np
    import time
    import tensorflow as tf
    import uuid
    
    # Use time-based seed + UUID hash to ensure different results on each run
    # This ensures diversity even if multiple calls happen in the same millisecond
    unique_id = str(uuid.uuid4())
    seed = (int(time.time() * 1000) + hash(unique_id)) % (2**31)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    parser = _build_parser()
    args = parser.parse_args()

    prompt_config = PromptConfig()
    deck_config = DeckConfig()

    # Use the prompt as-is (it's already formatted by the NestJS service)
    prompt_text = args.prompt.strip()

    prompt_vocab = _load_prompt_vocab(args.prompt_vocab)
    vectorizer = create_text_vectorizer_from_vocab(prompt_vocab, prompt_config=prompt_config)

    card_vocab = _load_card_vocab(args.card_vocab)
    card_to_index = card_vocab["card_to_index"]
    index_to_card = card_vocab["index_to_card"]

    tournament_priors = None
    if args.tournament_priors:
        try:
            with args.tournament_priors.open("r", encoding="utf-8") as fp:
                tournament_priors = json.load(fp)
        except FileNotFoundError:
            import sys
            print(f"WARNING: Tournament priors not found at {args.tournament_priors}", file=sys.stderr)
        except json.JSONDecodeError as exc:
            import sys
            print(f"WARNING: Failed to load priors from {args.tournament_priors}: {exc}", file=sys.stderr)

    # Load card repository and build card features if model uses them
    card_features = None
    try:
        repository = CardRepository(data_root=args.data_root, language="en")
        repository.load()
        # Create ability vectorizer
        ability_vectorizer = create_ability_vectorizer(repository, vocab_size=2000, max_length=64)
        # Build card feature matrix
        card_features = build_card_feature_matrix(
            card_to_index,
            repository,
            ability_vectorizer=ability_vectorizer,
            max_ability_length=64,
        )
        import sys
        print(f"DEBUG: Loaded card features for {len(card_to_index)} cards", file=sys.stderr)
    except Exception as e:
        import sys
        import warnings
        warnings.warn(f"Failed to load card features: {e}. Inference will continue without card features.")
        print(f"WARNING: Card features not available: {e}", file=sys.stderr)
        card_features = None

    # Build allowlist indices if provided
    allowlist_ids: set[int] = set()
    if args.first_token_allowlist:
        requested_ids = []
        found_ids = []
        for raw in args.first_token_allowlist.split(","):
            cid = raw.strip().upper()
            if not cid:
                continue
            requested_ids.append(cid)
            if cid in card_to_index:
                idx = int(card_to_index[cid])
                allowlist_ids.add(idx)
                found_ids.append(f"{cid} (idx={idx})")
            else:
                # Try base card ID without variant suffix
                base_id = cid.split("_")[0]
                if base_id in card_to_index:
                    idx = int(card_to_index[base_id])
                    allowlist_ids.add(idx)
                    found_ids.append(f"{base_id} (idx={idx}, from {cid})")
        
        if requested_ids and not allowlist_ids:
            import sys
            print(f"WARNING: None of the requested allowlist IDs were found in vocabulary: {requested_ids[:10]}...", file=sys.stderr)
            print(f"WARNING: This means the model cannot generate any of the requested leaders. Falling back to unrestricted generation.", file=sys.stderr)
        elif allowlist_ids:
            import sys
            print(f"Allowlist: {len(allowlist_ids)} IDs found from {len(requested_ids)} requested: {found_ids[:5]}", file=sys.stderr)
            if len(allowlist_ids) < len(requested_ids):
                missing_count = len(requested_ids) - len(allowlist_ids)
                print(f"WARNING: {missing_count} requested IDs were not found in model vocabulary and will be excluded from generation.", file=sys.stderr)

    if tournament_priors:
        if allowlist_ids:
            candidate_leaders = [
                index_to_card.get(int(idx))
                for idx in allowlist_ids
                if int(idx) in index_to_card
            ]
        else:
            candidate_leaders = list(tournament_priors.keys())
        control_tokens = _aggregate_prior_tokens(candidate_leaders, tournament_priors)
        if control_tokens:
            prompt_text = f"{control_tokens} {prompt_text}"

    # Enable unsafe deserialization to allow loading Lambda layers
    # (We trust our own trained models)
    import keras
    keras.config.enable_unsafe_deserialization()
    
    # Provide custom objects explicitly to ensure they can be deserialized
    custom_objects = {
        "TransformerEncoderLayer": TransformerEncoderLayer,
        "TransformerDecoderLayer": TransformerDecoderLayer,
        "PositionalEncodingLayer": PositionalEncodingLayer,
        "_create_combined_decoder_mask": _create_combined_decoder_mask,
        "gather_card_features": gather_card_features,
    }
    
    model = tf.keras.models.load_model(args.model, compile=False, custom_objects=custom_objects, safe_mode=False)
    
    # Check if model supports card features by inspecting input signature
    # Models with card features have 8 inputs, models without have 2 inputs
    model_input_count = len(model.inputs) if hasattr(model, 'inputs') and model.inputs else 0
    use_card_features_in_model = model_input_count >= 8
    
    if card_features and not use_card_features_in_model:
        import sys
        print(f"WARNING: Model was trained without card features (has {model_input_count} inputs), but card features are available. Ignoring card features.", file=sys.stderr)
        card_features = None
    elif not card_features and use_card_features_in_model:
        import sys
        print(f"WARNING: Model was trained with card features (has {model_input_count} inputs), but card features are not available. Model may not work correctly.", file=sys.stderr)

    def restrict_first_token(logits: tf.Tensor) -> tf.Tensor:
        # Always block special tokens (EOS, PAD, BOS) from being selected as the first token
        start_id = card_to_index[deck_config.start_token]
        end_id = card_to_index[deck_config.end_token]
        pad_id = card_to_index[deck_config.pad_token]
        special_ids_to_block = {start_id, end_id, pad_id}
        
        # Block special tokens
        for special_id in special_ids_to_block:
            logits = tf.tensor_scatter_nd_update(
                logits,
                [[special_id]],
                [tf.constant(-1e9, dtype=tf.float32)]
            )
        
        # If allowlist is provided, further restrict to only allowlisted tokens
        if allowlist_ids:
            # Create mask: set all non-allowlisted tokens to very negative
            mask = tf.ones_like(logits) * (-1e9)
            indices = tf.constant(sorted(list(allowlist_ids)), dtype=tf.int32)
            updates = tf.gather(logits, indices)
            logits = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), updates)
        else:
            # No allowlist - just block special tokens (already done above)
            pass
        
        return logits

    # Fast path: leader-only one-step decode
    if args.leader_only:
        prompt_tensor = vectorizer(tf.constant([prompt_text]))
        start_id = card_to_index[deck_config.start_token]
        pad_id = card_to_index[deck_config.pad_token]
        max_length = deck_config.max_total_cards + 2
        # Build decoder input of fixed length as expected by the model
        decoder_input_list = [start_id] + [pad_id] * (max_length - 1)
        decoder_input = tf.constant([decoder_input_list], dtype=tf.int32)
        # Prepare model inputs - only include card features if model supports them
        model_inputs = [prompt_tensor, decoder_input]
        if card_features and use_card_features_in_model:
            cost_feat = tf.expand_dims(card_features["cost"], 0) if len(card_features["cost"].shape) == 1 else card_features["cost"]
            power_feat = tf.expand_dims(card_features["power"], 0) if len(card_features["power"].shape) == 1 else card_features["power"]
            color_feat = tf.expand_dims(card_features["color"], 0) if len(card_features["color"].shape) == 1 else card_features["color"]
            type_feat = tf.expand_dims(card_features["type"], 0) if len(card_features["type"].shape) == 1 else card_features["type"]
            ability_feat = tf.expand_dims(card_features["ability_embedding"], 0) if len(card_features["ability_embedding"].shape) == 2 else card_features["ability_embedding"]
            has_ability_feat = tf.expand_dims(card_features["has_ability"], 0) if len(card_features["has_ability"].shape) == 1 else card_features["has_ability"]
            model_inputs.extend([cost_feat, power_feat, color_feat, type_feat, ability_feat, has_ability_feat])
        model_outputs = model(model_inputs, training=False)
        logits = model_outputs["main"] if isinstance(model_outputs, dict) else model_outputs
        first_step_logits = logits[0, 0]
        first_step_logits = restrict_first_token(first_step_logits)
        
        # If allowlist was provided, ensure we only sample from allowlisted tokens
        # After restriction, non-allowlisted tokens have logits of -1e9 (very negative)
        # So we should only consider tokens with logits > -1e8
        if allowlist_ids:
            # Count how many tokens have valid logits (not masked out)
            valid_logits_mask = first_step_logits > -1e8
            valid_count = tf.reduce_sum(tf.cast(valid_logits_mask, tf.int32))
            if valid_count == 0:
                import sys
                print(f"ERROR: All tokens were masked out by allowlist! This means no allowlisted IDs are in the model vocabulary.", file=sys.stderr)
                print(f"Allowlist had {len(allowlist_ids)} IDs, but none matched the vocabulary.", file=sys.stderr)
                # Fallback: return empty (will be handled by caller)
                leader_id = ""
            else:
                # Use temperature sampling for diversity instead of argmax
                # This ensures different leaders are suggested for the same prompt
                temperature = args.leader_temperature
                # Limit top_k to the number of valid (non-masked) tokens
                top_k = min(args.leader_top_k, int(valid_count.numpy()))
                
                if top_k == 0:
                    import sys
                    print(f"ERROR: No valid tokens after allowlist restriction!", file=sys.stderr)
                    leader_id = ""
                else:
                    # Apply temperature scaling
                    scaled_logits = first_step_logits / temperature
                    
                    # Get top-k candidates (these will only be from allowlisted tokens)
                    top_k_logits, top_k_indices = tf.nn.top_k(scaled_logits, k=top_k)
                    
                    # Convert to probabilities
                    probs = tf.nn.softmax(top_k_logits)
                    
                    # Sample from top-k instead of always picking the top one
                    # This introduces diversity while still preferring high-probability leaders
                    next_token_idx = tf.random.categorical(tf.expand_dims(probs, 0), 1)[0, 0]
                    next_token = int(top_k_indices[next_token_idx])
                    
                    leader_id = index_to_card.get(next_token, "")
                    if not leader_id:
                        import sys
                        print(f"WARNING: Generated token {next_token} is not in index_to_card mapping!", file=sys.stderr)
        else:
            # No allowlist - use original logic
            temperature = args.leader_temperature
            top_k = min(args.leader_top_k, len(first_step_logits))
            
            # Apply temperature scaling
            scaled_logits = first_step_logits / temperature
            
            # Get top-k candidates
            top_k_logits, top_k_indices = tf.nn.top_k(scaled_logits, k=top_k)
            
            # Convert to probabilities
            probs = tf.nn.softmax(top_k_logits)
            
            # Sample from top-k instead of always picking the top one
            # This introduces diversity while still preferring high-probability leaders
            next_token_idx = tf.random.categorical(tf.expand_dims(probs, 0), 1)[0, 0]
            next_token = int(top_k_indices[next_token_idx])
            
            leader_id = index_to_card.get(next_token, "")
        payload: Dict[str, object] = {"leader_id": leader_id, "main_deck": []}
        print(json.dumps(payload))
        return

    # Build initial sequence: [start_id, leader_id] if leader is specified, otherwise [start_id]
    initial_sequence: Optional[List[int]] = None
    if args.leader_id:
        leader_id_upper = args.leader_id.strip().upper()
        start_id = card_to_index[deck_config.start_token]
        if leader_id_upper in card_to_index:
            leader_token_id = card_to_index[leader_id_upper]
            initial_sequence = [start_id, leader_token_id]
            import sys
            print(f"DEBUG: Using initial sequence with leader: [start_id={start_id}, leader_id={leader_id_upper} (token={leader_token_id})]", file=sys.stderr)
        else:
            # Try base card ID without variant suffix
            base_id = leader_id_upper.split("_")[0]
            if base_id in card_to_index:
                leader_token_id = card_to_index[base_id]
                initial_sequence = [start_id, leader_token_id]
                import sys
                print(f"DEBUG: Using initial sequence with leader (base ID): [start_id={start_id}, leader_id={base_id} (token={leader_token_id})]", file=sys.stderr)
            else:
                import sys
                print(f"WARNING: Leader ID {leader_id_upper} not found in vocabulary, starting with just [start_id]", file=sys.stderr)

    if args.decode_strategy == "beam":
        top_n = max(1, args.rerank_beams)
        beam_results = beam_search_generate(
            prompt=prompt_text,
            model=model,
            prompt_vectorizer=vectorizer,
            card_to_index=card_to_index,
            index_to_card=index_to_card,
            deck_config=deck_config,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            initial_sequence=initial_sequence,
            card_features=card_features,
            use_card_features=use_card_features_in_model,
            repository=repository,
            tournament_priors=tournament_priors,
            return_top_n=top_n,
        )

        if isinstance(beam_results, list) and beam_results:
            if isinstance(beam_results[0], list):
                candidate_sequences = beam_results  # type: ignore[assignment]
            else:
                candidate_sequences = [beam_results]  # type: ignore[list-item]
        else:
            candidate_sequences = []

        if args.rerank_beams > 1 and candidate_sequences:
            sequence = rerank_candidates(
                prompt=prompt_text,
                candidates=candidate_sequences,
                deck_config=deck_config,
                repository=repository,
                priors=tournament_priors,
            )
        else:
            sequence = candidate_sequences[0] if candidate_sequences else []
    else:
        if args.rerank_beams > 1:
            import sys
            print(
                "WARNING: --rerank-beams is only supported with beam decode. Ignoring parameter.",
                file=sys.stderr,
            )
        # Use temperature sampling with EOS penalty to prevent early termination
        # Temperature 1.3 allows some diversity while staying close to model predictions
        # Top-k 50 limits to reasonable card predictions
        # EOS penalty 5.0 discourages ending too early (when we have < 10 tokens)
        sequence = greedy_generate(
            prompt=prompt_text,
            model=model,
            prompt_vectorizer=vectorizer,
            card_to_index=card_to_index,
            index_to_card=index_to_card,
            deck_config=deck_config,
            initial_sequence=initial_sequence,
            temperature=0.1,  # Very low temperature to make selection almost deterministic, strongly favor biased cards
            top_k=10,  # Reduced from 15 - focus on top predictions, allows bias to have more impact
            eos_penalty=5.0,
            card_features=card_features,
            use_card_features=use_card_features_in_model,
            repository=repository,
            tournament_priors=tournament_priors,
        )

    if not sequence:
        # Return a valid JSON response instead of raising SystemExit
        # This allows the NestJS service to handle the empty result gracefully
        payload: Dict[str, object] = {
            "leader_id": args.leader_id if args.leader_id else "",
            "main_deck": [],
        }
        print(json.dumps(payload))
        return  # Exit normally with code 0

    leader_id = sequence[0] if sequence else ""
    main_deck = sequence[1 : 1 + deck_config.main_deck_size]
    sideboard = sequence[1 + deck_config.main_deck_size :]

    if leader_id:
        main_deck = repair_deck(
            leader_id=leader_id,
            main_deck=main_deck,
            prompt=prompt_text,
            deck_config=deck_config,
            repository=repository,
        )

    payload: Dict[str, object] = {
        "leader_id": leader_id,
        "main_deck": main_deck,
    }
    if sideboard:
        payload["sideboard"] = sideboard

    print(json.dumps(payload))


if __name__ == "__main__":
    main()

