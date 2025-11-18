"""
End-to-end training script for the deck transformer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tensorflow as tf

from ..config import DeckConfig, PromptConfig
from ..data import CardRepository, PromptDeckExample
from ..datasets import (
    build_card_vocabulary,
    create_text_vectorizer,
    load_examples_jsonl,
    make_tf_dataset,
)
from ..models import TransformerConfig, build_deck_transformer
from .losses import masked_accuracy, masked_sparse_categorical_crossentropy
from .custom_training_step import AutoregressiveSequenceLossStep


def configure_tensorflow_for_gpu() -> None:
    """
    Configure TensorFlow for optimal GPU performance.
    
    Enables GPU memory growth and optimizes TensorFlow settings for GPU-based training.
    Falls back to CPU configuration if no GPU is detected.
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for faster training (optional, can be enabled if needed)
            # tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            print(f"TensorFlow GPU configuration:")
            print(f"  GPUs detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"    GPU {i}: {gpu.name}")
            print(f"  Memory growth: Enabled")
            return
    except Exception as e:
        print(f"GPU configuration failed: {e}, falling back to CPU")
    
    # Fallback to CPU configuration
    configure_tensorflow_for_cpu()


def configure_tensorflow_for_cpu() -> None:
    """
    Configure TensorFlow for optimal CPU performance.
    
    Sets thread counts, disables GPU if not needed, and optimizes
    TensorFlow settings for CPU-based training.
    """
    num_cores = os.cpu_count() or 8
    
    # Calculate optimal thread distribution
    # inter_op: threads for parallel operations (keep low to avoid overhead)
    # intra_op: threads within a single operation (use most cores)
    inter_threads = min(4, max(2, num_cores // 4))
    intra_threads = max(1, num_cores - inter_threads)
    
    # Set environment variables (must be set before TensorFlow operations)
    os.environ['TF_NUM_INTEROP_THREADS'] = str(inter_threads)
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(intra_threads)
    os.environ['OMP_NUM_THREADS'] = str(intra_threads)
    
    # Configure TensorFlow threading
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    
    # Disable GPU if not needed (prevents GPU initialization overhead on CPU-only systems)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            tf.config.set_visible_devices([], 'GPU')
            print("No GPU detected, using CPU only.")
    except Exception:
        pass  # GPU config may fail if no GPU is present
    
    # Optimize for CPU
    try:
        # Disable JIT compilation which can sometimes slow CPU training
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass  # JIT setting may not be available in all TensorFlow versions
    
    print(f"TensorFlow CPU configuration:")
    print(f"  CPU cores detected: {num_cores}")
    print(f"  Inter-op threads (parallel operations): {inter_threads}")
    print(f"  Intra-op threads (within operations): {intra_threads}")
    print(f"  OMP_NUM_THREADS: {intra_threads}")


def configure_process_priority() -> None:
    """
    Configure process priority and CPU affinity for better performance.
    
    This is optional and requires psutil. If psutil is not available,
    this function will silently skip the optimizations.
    """
    try:
        import psutil
        import sys
        
        # Set process priority to high (Windows)
        try:
            p = psutil.Process(os.getpid())
            if sys.platform == 'win32':
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                print("  Process priority set to HIGH")
            else:
                p.nice(-5)  # Higher priority on Unix-like systems
                print("  Process priority increased")
        except (psutil.AccessDenied, AttributeError):
            # May require admin privileges or not be available
            pass
        
        # Set CPU affinity (optional - can help on systems with many cores)
        # Uncomment and adjust if you want to pin to specific cores
        # try:
        #     p = psutil.Process(os.getpid())
        #     num_cores = os.cpu_count() or 8
        #     # Use first N cores (adjust as needed)
        #     cores_to_use = list(range(min(8, num_cores)))
        #     p.cpu_affinity(cores_to_use)
        #     print(f"  CPU affinity set to cores: {cores_to_use}")
        # except (psutil.AccessDenied, AttributeError):
        #     pass
            
    except ImportError:
        # psutil not installed - skip process optimizations
        pass


def _split_examples(examples, split_name: str) -> List:
    return [example for example in examples if (example.split or "train") == split_name]


def _compute_sample_weights(examples: Sequence[PromptDeckExample]) -> List[float]:
    weights: List[float] = []
    for example in examples:
        weight = 1.0
        tags = example.quality_tags or []
        if "winner" in tags:
            weight = 2.0
        elif "top-3" in tags:
            weight = 1.8
        elif "top-16" in tags:
            weight = 1.5
        elif "tournament" in tags:
            weight = 1.3
        weights.append(weight)
    return weights


class EpochTracker(tf.keras.callbacks.Callback):
    """
    Persists the number of completed epochs so training can resume after interruptions.
    """

    def __init__(self, state_path: Path) -> None:
        super().__init__()
        self._state_path = state_path

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        state = {"completed_epochs": int(epoch) + 1}
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_path.open("w", encoding="utf-8") as fp:
            json.dump(state, fp, indent=2)


def prepare_datasets(
    dataset_path: Path,
    deck_config: DeckConfig,
    prompt_config: PromptConfig,
    batch_size: int,
    seed: int,
    use_card_features: bool = True,
    data_root: Path = Path("data"),
    include_control_tokens: bool = True,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    Optional[tf.data.Dataset],
    Dict[str, int],
    List[str],
    Optional[Dict[str, tf.Tensor]],
]:
    examples = load_examples_jsonl(dataset_path)
    train_examples = _split_examples(examples, "train")
    val_examples = _split_examples(examples, "val")
    test_examples = _split_examples(examples, "test")

    if not val_examples:
        val_examples = train_examples[-max(1, len(train_examples) // 10) :]
        train_examples = train_examples[: -len(val_examples)]

    prompt_vectorizer = create_text_vectorizer(
        [example.prompt for example in train_examples],
        prompt_config=prompt_config,
    )
    card_to_index, index_to_card = build_card_vocabulary(train_examples, deck_config=deck_config)

    # Load card repository for card features
    repository = None
    if use_card_features:
        try:
            repository = CardRepository(data_root=data_root, language="en")
            repository.load()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load card repository: {e}. Continuing without card features.")
            use_card_features = False

    train_ds, train_card_features = make_tf_dataset(
        train_examples,
        prompt_vectorizer=prompt_vectorizer,
        card_to_index=card_to_index,
        deck_config=deck_config,
        prompt_config=prompt_config,
        batch_size=batch_size,
        seed=seed,
        data_root=data_root,
        repository=repository,
        use_card_features=use_card_features,
        example_weights=_compute_sample_weights(train_examples),
        include_control_tokens=include_control_tokens,
    )
    val_ds, val_card_features = make_tf_dataset(
        val_examples,
        prompt_vectorizer=prompt_vectorizer,
        card_to_index=card_to_index,
        deck_config=deck_config,
        prompt_config=prompt_config,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        prompt_dropout_rate=0.0,
        prompt_paraphrase_rate=0.0,
        deck_shuffle_rate=0.0,
        data_root=data_root,
        repository=repository,
        use_card_features=use_card_features,
        example_weights=[1.0] * len(val_examples),
        include_control_tokens=include_control_tokens,
    )
    if test_examples:
        test_ds, test_card_features = make_tf_dataset(
            test_examples,
            prompt_vectorizer=prompt_vectorizer,
            card_to_index=card_to_index,
            deck_config=deck_config,
            prompt_config=prompt_config,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            prompt_dropout_rate=0.0,
            prompt_paraphrase_rate=0.0,
            deck_shuffle_rate=0.0,
            data_root=data_root,
            repository=repository,
            use_card_features=use_card_features,
            example_weights=[1.0] * len(test_examples),
            include_control_tokens=include_control_tokens,
        )
    else:
        test_ds = None
        test_card_features = None

    # Use train_card_features as the canonical card features (they should be the same)
    card_features = train_card_features

    prompt_vocab = prompt_vectorizer.get_vocabulary()
    return train_ds, val_ds, test_ds, card_to_index, prompt_vocab, card_features


def train(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 1234,
    transformer_config: TransformerConfig = TransformerConfig(),
    early_stopping_patience: Optional[int] = 3,
    include_control_tokens: bool = True,
    resume_from: Optional[Path] = None,
    freq_hist_weight: float = 100.0,  # Increased from 10.0 - need much stronger penalty to match tournament decks (16.3 unique cards, 2.1 at 1x)
    entropy_penalty: float = 2.0,  # Increased from 1.0 - encourage more concentration on fewer cards
    low_prob_penalty: float = 10.0,  # Increased from 5.0 - stronger penalty for 1x cards
    low_prob_threshold: float = 0.3,  # Increased from 0.25 - penalize more cards
    sequence_level_weight: float = 200.0,  # Phase 1: Weight for sequence-level loss - Increased from 100.0 to 200.0 to strongly penalize 49 unique cards (with new penalties: 49 unique = ~10.7M loss contribution)
    use_phase2_autoregressive: bool = True,  # Phase 2: Use autoregressive sequence-level loss (generates sequences during training)
    phase2_scheduled_sampling_rate: float = 0.5,  # Phase 2: Probability of using model's own predictions vs teacher forcing
    phase2_generation_batch_fraction: float = 0.25,  # Phase 2: Fraction of batch to generate autoregressively (to save compute)
    save_checkpoints: bool = True,
) -> None:
    deck_config = DeckConfig()
    prompt_config = PromptConfig()

    train_ds, val_ds, test_ds, card_to_index, prompt_vocab, card_features = prepare_datasets(
        dataset_path,
        deck_config=deck_config,
        prompt_config=prompt_config,
        batch_size=batch_size,
        seed=seed,
        use_card_features=True,
        include_control_tokens=include_control_tokens,
    )

    pad_token_id = card_to_index[deck_config.pad_token]
    start_token_id = card_to_index[deck_config.start_token]
    end_token_id = card_to_index[deck_config.end_token]
    use_card_features = card_features is not None
    model = build_deck_transformer(
        prompt_vocab_size=len(prompt_vocab),
        deck_vocab_size=len(card_to_index),
        prompt_sequence_length=prompt_config.max_length,
        deck_sequence_length=deck_config.max_total_cards + 2,
        pad_token_id=pad_token_id,
        config=transformer_config,
        use_card_features=use_card_features,
        start_token_id=start_token_id,  # Phase 1: For sequence-level loss
        end_token_id=end_token_id,  # Phase 1: For sequence-level loss
    )

     # Log model input information
    num_inputs = len(model.inputs)
    print(f"Model built with {num_inputs} inputs (card features: {'enabled' if use_card_features else 'disabled'})")
    if use_card_features:
        print(f"  Input names: {[inp.name for inp in model.inputs]}")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )

    loss_fn = masked_sparse_categorical_crossentropy(pad_token_id=pad_token_id)
    acc_fn = masked_accuracy(pad_token_id=pad_token_id)

    from .losses import anti_singleton_loss, sequence_level_loss
    
    # Get special token IDs for sequence-level loss
    start_token_id = card_to_index[deck_config.start_token]
    end_token_id = card_to_index[deck_config.end_token]
    
    losses = {
        "main": loss_fn,
        "type_aux": tf.keras.losses.KLDivergence(name="type_kld"),
        "cost_aux": tf.keras.losses.KLDivergence(name="cost_kld"),
        "freq_hist": anti_singleton_loss(
            mse_weight=0.1,  # Keep some MSE for distribution matching
            entropy_penalty=entropy_penalty,  # Penalize high entropy (encourages concentration)
            low_prob_penalty=low_prob_penalty,  # Strongly penalize low-probability cards (1x cards)
            low_prob_threshold=low_prob_threshold,
        ),
        # Phase 1: Sequence-level loss - directly penalizes singleton-heavy generated sequences
        "predicted_sequence_freq_hist": sequence_level_loss(
            pad_token_id=pad_token_id,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            max_copies=deck_config.max_copies_per_card,
        ),
    }
    # Increased freq_hist weight from 0.1 to 1.0 (default) to strongly encourage realistic card counts
    # This helps the model learn that 4x staples and 2x tech cards are more common than 1x cards
    # Higher weight (1.0+) significantly reduces the number of 1x cards in generated decks
    # Phase 1: Add sequence-level loss weight (default 50.0) to directly penalize singleton-heavy generations
    loss_weights = {
        "main": 1.0,
        "type_aux": 0.2,
        "cost_aux": 0.2,
        "freq_hist": freq_hist_weight,
        "predicted_sequence_freq_hist": sequence_level_weight,  # Phase 1: Sequence-level loss weight
    }
    metrics = {
        "main": [acc_fn],
        "freq_hist": [tf.keras.metrics.MeanSquaredError(name="freq_hist_metric")],
    }

    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
    
    # Phase 2: Wrap model with custom training step for autoregressive sequence-level loss
    index_to_card = {index: card_id for card_id, index in card_to_index.items()}
    if use_phase2_autoregressive:
        print("Phase 2: Using autoregressive sequence-level loss (generates sequences during training)")
        print(f"  Scheduled sampling rate: {phase2_scheduled_sampling_rate}")
        print(f"  Generation batch fraction: {phase2_generation_batch_fraction}")
        model = AutoregressiveSequenceLossStep(
            base_model=model,
            sequence_level_loss_fn=sequence_level_loss(
                pad_token_id=pad_token_id,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                max_copies=deck_config.max_copies_per_card,
            ),
            sequence_level_weight=sequence_level_weight,
            card_to_index=card_to_index,
            index_to_card=index_to_card,
            deck_config=deck_config,
            use_card_features=use_card_features,
            scheduled_sampling_rate=phase2_scheduled_sampling_rate,
            generation_batch_fraction=phase2_generation_batch_fraction,
            losses=losses,
            loss_weights=loss_weights,
        )
        # Recompile with the wrapped model (it will use the base model's compiled losses/metrics)
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
    
    # Wrap datasets to include card features if enabled
    if use_card_features and card_features:
        def add_card_features(x, y, sample_weight):
            # x is a tuple of (prompt_tokens, decoder_input)
            # y is decoder_target
            prompt_tokens, decoder_input = x
            # Card features are constant tensors, broadcast them for each batch
            # We need to tile them to match batch size, but actually they're the same for all batches
            # So we can just pass them as-is (they'll be broadcast automatically)
            # Create constant tensors for card features (they're the same for all batches)
            cost_feat = tf.expand_dims(card_features["cost"], 0)  # (1, vocab_size)
            power_feat = tf.expand_dims(card_features["power"], 0)  # (1, vocab_size)
            color_feat = tf.expand_dims(card_features["color"], 0)  # (1, vocab_size)
            type_feat = tf.expand_dims(card_features["type"], 0)  # (1, vocab_size)
            ability_feat = tf.expand_dims(card_features["ability_embedding"], 0)  # (1, vocab_size, 64)
            has_ability_feat = tf.expand_dims(card_features["has_ability"], 0)  # (1, vocab_size)
            
            # Tile to match batch size (though broadcasting should work)
            # Actually, we don't need to tile - the model will handle broadcasting
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
        
        train_ds = train_ds.map(add_card_features, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(add_card_features, num_parallel_calls=tf.data.AUTOTUNE)
        if test_ds:
            test_ds = test_ds.map(add_card_features, num_parallel_calls=tf.data.AUTOTUNE)

    resuming = resume_from is not None
    if resuming:
        run_dir = resume_from
        print(f"Resuming training from {run_dir}")
    else:
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "logs"
    vocab_dir = run_dir / "vocab"
    state_path = run_dir / "training_state.json"
    history_path = run_dir / "history.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    if not resuming:
        vocab_dir.mkdir(parents=True, exist_ok=True)
        with (vocab_dir / "prompt_vocabulary.txt").open("w", encoding="utf-8") as fp:
            for token in prompt_vocab:
                fp.write(token + "\n")
        index_to_card = {index: card_id for card_id, index in card_to_index.items()}
        with (vocab_dir / "card_vocabulary.json").open("w", encoding="utf-8") as fp:
            json.dump({"card_to_index": card_to_index, "index_to_card": index_to_card}, fp, indent=2)
    else:
        vocab_dir.mkdir(parents=True, exist_ok=True)

    previous_history: Dict[str, List[float]] = {}
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as fp:
            loaded_history = json.load(fp)
            previous_history = {
                key: [float(v) for v in values] for key, values in loaded_history.items()
            }

    initial_epoch = 0
    if resuming:
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as fp:
                state = json.load(fp)
                initial_epoch = int(state.get("completed_epochs", 0))
        elif previous_history.get("loss"):
            initial_epoch = len(previous_history["loss"])

    if resuming:
        if not save_checkpoints:
            print("WARNING: Resuming training but checkpoints are disabled. Cannot load weights; starting from scratch.")
        else:
            latest_ckpt = checkpoints_dir / "ckpt_latest.weights.h5"
            fallback_ckpts = sorted(checkpoints_dir.glob("ckpt_*.weights.h5"))
            resume_weights_path = None
            if latest_ckpt.exists():
                resume_weights_path = latest_ckpt
            elif fallback_ckpts:
                resume_weights_path = fallback_ckpts[-1]
            if resume_weights_path and resume_weights_path.exists():
                model.load_weights(str(resume_weights_path))
                print(f"Loaded weights from {resume_weights_path}")
            else:
                print("No checkpoint weights found; starting from randomly initialized weights.")

    if initial_epoch >= epochs:
        print(
            f"Completed epochs ({initial_epoch}) are >= requested epochs ({epochs}); "
            "skipping training."
        )
        return

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)),
        EpochTracker(state_path=state_path),
    ]
    
    # Only add checkpoint callbacks if enabled (saves disk space on cloud training)
    if save_checkpoints:
        callbacks.extend([
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoints_dir / "ckpt_latest.weights.h5"),
                save_weights_only=True,
                save_best_only=False,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoints_dir / "ckpt_{epoch:02d}.weights.h5"),
                save_weights_only=True,
                monitor="val_loss",
                save_best_only=True,
            ),
        ])
    else:
        print("Checkpoints disabled - only final model will be saved (saves disk space)")
    
    if early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    if previous_history:
        combined_history: Dict[str, List[float]] = {}
        for key in set(previous_history.keys()) | set(history_dict.keys()):
            combined_history[key] = previous_history.get(key, []) + history_dict.get(key, [])
    else:
        combined_history = history_dict

    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(combined_history, fp, indent=2)

    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, return_dict=True)
        with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as fp:
            json.dump({k: float(v) for k, v in test_metrics.items()}, fp, indent=2)

    model.save(run_dir / "deck_transformer.keras")
    print(f"Training run complete. Artifacts saved to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deck transformer model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("ml/artifacts/synthetic_prompt_deck.jsonl"),
        help="Path to JSONL dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory for run artifacts.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--coverage-weight", type=float, default=0.01)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of epochs with no improvement after which training will be stopped. Set to 0 or negative to disable early stopping.",
    )
    parser.add_argument(
        "--disable-control-tokens",
        action="store_true",
        help="Disable control token prefixes in training prompts.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Path to an existing run directory to continue training from.",
    )
    parser.add_argument(
        "--freq-hist-weight",
        type=float,
        default=100.0,  # Increased from 10.0 - need much stronger penalty to match tournament decks (16.3 unique cards, 2.1 at 1x)
        help="Weight for the frequency histogram loss (higher = stronger regularization against 1x cards). Default: 100.0",
    )
    parser.add_argument(
        "--entropy-penalty",
        type=float,
        default=2.0,  # Increased from 1.0 - encourage more concentration on fewer cards
        help="Weight for entropy penalty in freq_hist loss (higher = more concentration on fewer cards). Default: 2.0",
    )
    parser.add_argument(
        "--low-prob-penalty",
        type=float,
        default=10.0,  # Increased from 5.0 - stronger penalty for 1x cards
        help="Weight for low-probability penalty in freq_hist loss (higher = stronger penalty for 1x cards). Default: 10.0",
    )
    parser.add_argument(
        "--low-prob-threshold",
        type=float,
        default=0.3,  # Increased from 0.25 - penalize more cards
        help="Threshold for considering a card 'low probability' in freq_hist loss (matches 1 copy when hist targets are scaled by max copies). Default: 0.3",
    )
    parser.add_argument(
        "--sequence-level-weight",
        type=float,
        default=200.0,  # Phase 1: Sequence-level loss weight
        help="Weight for the sequence-level loss (Phase 1: directly penalizes singleton-heavy generated sequences). Default: 200.0",
    )
    parser.add_argument(
        "--disable-phase2",
        action="store_true",
        help="Disable Phase 2 autoregressive sequence-level loss (use only Phase 1). Default: Phase 2 enabled",
    )
    parser.add_argument(
        "--phase2-scheduled-sampling-rate",
        type=float,
        default=0.5,
        help="Phase 2: Probability of using model's own predictions vs teacher forcing during autoregressive generation. Default: 0.5",
    )
    parser.add_argument(
        "--phase2-generation-batch-fraction",
        type=float,
        default=0.25,
        help="Phase 2: Fraction of batch to generate autoregressively (to save compute). Default: 0.25",
    )
    parser.add_argument(
        "--disable-checkpoints",
        action="store_true",
        help="Disable saving checkpoints during training (saves disk space, useful for cloud training like RunPod). Only the final model will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    # Configure TensorFlow for GPU (falls back to CPU if no GPU detected)
    configure_tensorflow_for_gpu()
    configure_process_priority()
    
    args = parse_args()
    config = TransformerConfig(
        embedding_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        coverage_loss_weight=args.coverage_weight,
    )
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None
    train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        transformer_config=config,
        early_stopping_patience=early_stopping_patience,
        include_control_tokens=not args.disable_control_tokens,
        resume_from=args.resume_from,
        freq_hist_weight=args.freq_hist_weight,
        entropy_penalty=args.entropy_penalty,
        low_prob_penalty=args.low_prob_penalty,
        low_prob_threshold=args.low_prob_threshold,
        sequence_level_weight=args.sequence_level_weight,
        use_phase2_autoregressive=not args.disable_phase2,
        phase2_scheduled_sampling_rate=args.phase2_scheduled_sampling_rate,
        phase2_generation_batch_fraction=args.phase2_generation_batch_fraction,
        save_checkpoints=not args.disable_checkpoints,
    )


if __name__ == "__main__":
    main()

