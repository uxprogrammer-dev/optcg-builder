"""
Targeted fine-tune pass that only updates the duplicate-related heads.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import tensorflow as tf

from ..config import DeckConfig, PromptConfig
from ..models.transformer import (
    CombinedDecoderMaskLayer,
    ExtractFeatureLayer,
    PaddingMaskLayer,
    PositionalEncodingLayer,
    PredictedSequenceFreqHistLayer,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    gather_card_features,
)
from .losses import (
    anti_singleton_loss,
    masked_accuracy,
    masked_sparse_categorical_crossentropy,
    sequence_level_loss,
)
from .train import configure_tensorflow_for_gpu, prepare_datasets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the duplicate histogram heads on top of a trained model."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Path to the .keras model to start from (e.g., models/run_xxx/deck_transformer.keras).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/training/decks_merged.jsonl"),
        help="JSONL dataset to use for fine-tuning (ideally tournament-heavy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where the fine-tuned run should be written.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--freq-hist-weight",
        type=float,
        default=5.0,
        help="Weight for freq_hist loss during fine-tune.",
    )
    parser.add_argument(
        "--sequence-level-weight",
        type=float,
        default=10.0,
        help="Weight for predicted_sequence_freq_hist loss during fine-tune.",
    )
    parser.add_argument(
        "--main-loss-weight",
        type=float,
        default=0.05,
        help="Weight for the main decoder loss (kept small so histogram heads dominate).",
    )
    parser.add_argument(
        "--trainable-decoder-layers",
        type=int,
        default=1,
        help="Number of final decoder layers to unfreeze (default: only the last one).",
    )
    parser.add_argument(
        "--include-control-tokens",
        action="store_true",
        help="Match training CLI flag; normally leave enabled.",
    )
    return parser.parse_args()


def _load_model(model_path: Path) -> tf.keras.Model:
    import keras

    keras.config.enable_unsafe_deserialization()
    custom_objects = {
        "ExtractFeatureLayer": ExtractFeatureLayer,
        "TransformerEncoderLayer": TransformerEncoderLayer,
        "TransformerDecoderLayer": TransformerDecoderLayer,
        "PredictedSequenceFreqHistLayer": PredictedSequenceFreqHistLayer,
        "PositionalEncodingLayer": PositionalEncodingLayer,
        "PaddingMaskLayer": PaddingMaskLayer,
        "CombinedDecoderMaskLayer": CombinedDecoderMaskLayer,
        "gather_card_features": gather_card_features,
    }
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=custom_objects,
        safe_mode=False,
    )


def _collect_decoder_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    decoder_layers = [layer for layer in model.layers if layer.name.startswith("decoder_layer_")]
    decoder_layers.sort(key=lambda l: int(l.name.rsplit("_", 1)[-1]))
    return decoder_layers


def _set_trainable_layers(
    model: tf.keras.Model,
    decoder_layers_to_unfreeze: int,
    extra_layer_names: Iterable[str],
) -> None:
    decoder_layers = _collect_decoder_layers(model)
    trainable_layer_names = set(extra_layer_names)
    if decoder_layers_to_unfreeze > 0:
        for layer in decoder_layers[-decoder_layers_to_unfreeze:]:
            trainable_layer_names.add(layer.name)

    for layer in model.layers:
        layer.trainable = layer.name in trainable_layer_names

    trainable_layers = [layer.name for layer in model.layers if layer.trainable]
    print("Trainable layers for fine-tune:")
    for name in trainable_layers:
        print(f"  - {name}")


def _attach_card_features(
    ds: tf.data.Dataset, card_features: Dict[str, tf.Tensor]
) -> tf.data.Dataset:
    def add_card_features(x, y, sample_weight):
        prompt_tokens, decoder_input = x
        cost_feat = tf.expand_dims(card_features["cost"], 0)
        power_feat = tf.expand_dims(card_features["power"], 0)
        color_feat = tf.expand_dims(card_features["color"], 0)
        type_feat = tf.expand_dims(card_features["type"], 0)
        ability_feat = tf.expand_dims(card_features["ability_embedding"], 0)
        has_ability_feat = tf.expand_dims(card_features["has_ability"], 0)
        return (
            prompt_tokens,
            decoder_input,
            cost_feat,
            power_feat,
            color_feat,
            type_feat,
            ability_feat,
            has_ability_feat,
        ), y, sample_weight

    return ds.map(add_card_features, num_parallel_calls=tf.data.AUTOTUNE)


def _copy_vocab(base_model_path: Path, target_dir: Path) -> None:
    base_dir = base_model_path.parent
    vocab_dir = base_dir / "vocab"
    if not vocab_dir.exists():
        return
    target_vocab = target_dir / "vocab"
    if target_vocab.exists():
        return
    shutil.copytree(vocab_dir, target_vocab)


def main() -> None:
    args = _parse_args()
    configure_tensorflow_for_gpu()

    deck_config = DeckConfig()
    prompt_config = PromptConfig()

    train_ds, val_ds, _, card_to_index, _, card_features = prepare_datasets(
        dataset_path=args.dataset,
        deck_config=deck_config,
        prompt_config=prompt_config,
        batch_size=args.batch_size,
        seed=args.seed,
        use_card_features=True,
        include_control_tokens=args.include_control_tokens,
    )

    use_card_features = card_features is not None
    if use_card_features:
        train_ds = _attach_card_features(train_ds, card_features)
        val_ds = _attach_card_features(val_ds, card_features)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = _load_model(args.base_model)
    _set_trainable_layers(
        model,
        decoder_layers_to_unfreeze=max(1, args.trainable_decoder_layers),
        extra_layer_names={"freq_hist", "predicted_sequence_freq_hist"},
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    pad_token_id = card_to_index[deck_config.pad_token]
    start_token_id = card_to_index[deck_config.start_token]
    end_token_id = card_to_index[deck_config.end_token]

    losses = {
        "main": masked_sparse_categorical_crossentropy(pad_token_id=pad_token_id),
        "freq_hist": anti_singleton_loss(
            mse_weight=0.1,
            entropy_penalty=5.0,
            low_prob_penalty=20.0,
            low_prob_threshold=0.3,
        ),
        "predicted_sequence_freq_hist": sequence_level_loss(
            pad_token_id=pad_token_id,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            max_copies=deck_config.max_copies_per_card,
        ),
        "type_aux": tf.keras.losses.KLDivergence(name="type_kld"),
        "cost_aux": tf.keras.losses.KLDivergence(name="cost_kld"),
    }

    loss_weights = {
        "main": args.main_loss_weight,
        "freq_hist": args.freq_hist_weight,
        "predicted_sequence_freq_hist": args.sequence_level_weight,
        "type_aux": 0.05,
        "cost_aux": 0.05,
    }

    metrics = {
        "main": [masked_accuracy(pad_token_id=pad_token_id)],
        "freq_hist": [tf.keras.metrics.MeanSquaredError(name="freq_hist_metric")],
    }

    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"run_finetune_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _copy_vocab(args.base_model, run_dir)

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    history_path = run_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, fp, indent=2)

    model.save(run_dir / "deck_transformer.keras")
    print(f"Fine-tuned model saved to {run_dir}")


if __name__ == "__main__":
    main()

