"""
End-to-end training script for the intent classifier.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from ..config import PromptConfig
from ..data import CardRepository
from ..data.intent_schema import IntentExample
from ..datasets.intent_data import load_intent_examples
from ..datasets.intent_tfdata import build_intent_vocabularies, make_intent_dataset
from ..datasets.tfdata import create_text_vectorizer
from tensorflow.keras.layers import TextVectorization
from ..models.intent_classifier import IntentClassifierConfig, build_intent_classifier


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


def _split_examples(examples: List[IntentExample], split_name: str) -> List[IntentExample]:
    """Filter examples by split."""
    return [ex for ex in examples if (ex.split or "train") == split_name]


def prepare_datasets(
    dataset_path: Path,
    prompt_config: PromptConfig,
    batch_size: int,
    seed: int,
    data_root: Path = Path("data"),
    keyword_vocab_size: int = 1000,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    Optional[tf.data.Dataset],
    Dict[str, int],
    Dict[int, str],
    Dict[str, int],
    Dict[int, str],
    Dict[str, int],
    Dict[int, str],
    Dict[str, int],
    Dict[int, str],
    List[str],
    TextVectorization,
]:
    """
    Prepare datasets for intent classification training.
    
    Returns:
        Tuple of (train_ds, val_ds, test_ds, color_to_index, index_to_color,
                  keyword_to_index, index_to_keyword, leader_to_index, index_to_leader,
                  strategy_to_index, index_to_strategy, leader_ids, prompt_vectorizer)
    """
    examples = load_intent_examples(dataset_path)
    train_examples = _split_examples(examples, "train")
    val_examples = _split_examples(examples, "val")
    test_examples = _split_examples(examples, "test")

    if not val_examples:
        val_examples = train_examples[-max(1, len(train_examples) // 10) :]
        train_examples = train_examples[: -len(val_examples)]

    # Get all leader IDs from knowledge base
    repository = CardRepository(data_root=data_root, language="en")
    repository.load()
    leader_ids = [leader.canonical_id for leader in repository.leaders()]

    # Build vocabularies
    (
        color_to_index,
        index_to_color,
        keyword_to_index,
        index_to_keyword,
        leader_to_index,
        index_to_leader,
        strategy_to_index,
        index_to_strategy,
    ) = build_intent_vocabularies(examples, leader_ids, keyword_vocab_size=keyword_vocab_size)

    # Build prompt vectorizer
    prompt_vectorizer = create_text_vectorizer(
        [ex.prompt for ex in train_examples],
        prompt_config=prompt_config,
    )

    # Create datasets
    train_ds = make_intent_dataset(
        train_examples,
        prompt_vectorizer=prompt_vectorizer,
        color_to_index=color_to_index,
        keyword_to_index=keyword_to_index,
        leader_to_index=leader_to_index,
        strategy_to_index=strategy_to_index,
        prompt_config=prompt_config,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    val_ds = make_intent_dataset(
        val_examples,
        prompt_vectorizer=prompt_vectorizer,
        color_to_index=color_to_index,
        keyword_to_index=keyword_to_index,
        leader_to_index=leader_to_index,
        strategy_to_index=strategy_to_index,
        prompt_config=prompt_config,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )

    test_ds = None
    if test_examples:
        test_ds = make_intent_dataset(
            test_examples,
            prompt_vectorizer=prompt_vectorizer,
            color_to_index=color_to_index,
            keyword_to_index=keyword_to_index,
            leader_to_index=leader_to_index,
            strategy_to_index=strategy_to_index,
            prompt_config=prompt_config,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        )

    return (
        train_ds,
        val_ds,
        test_ds,
        color_to_index,
        index_to_color,
        keyword_to_index,
        index_to_keyword,
        leader_to_index,
        index_to_leader,
        strategy_to_index,
        index_to_strategy,
        leader_ids,
        prompt_vectorizer,
    )


def train(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 1234,
    config: IntentClassifierConfig = None,
    early_stopping_patience: Optional[int] = 3,
    keyword_vocab_size: int = 1000,
    data_root: Path = Path("data"),
) -> None:
    """Train the intent classifier model."""
    if config is None:
        config = IntentClassifierConfig()

    prompt_config = PromptConfig()

    (
        train_ds,
        val_ds,
        test_ds,
        color_to_index,
        index_to_color,
        keyword_to_index,
        index_to_keyword,
        leader_to_index,
        index_to_leader,
        strategy_to_index,
        index_to_strategy,
        leader_ids,
        prompt_vectorizer,
    ) = prepare_datasets(
        dataset_path,
        prompt_config=prompt_config,
        batch_size=batch_size,
        seed=seed,
        keyword_vocab_size=keyword_vocab_size,
        data_root=data_root,
    )

    # Build model
    model = build_intent_classifier(
        prompt_vocab_size=len(prompt_vectorizer.get_vocabulary()),
        prompt_max_length=prompt_config.max_length,
        num_colors=len(color_to_index),
        num_keywords=len(keyword_to_index),
        num_leaders=len(leader_to_index),
        num_strategies=len(strategy_to_index),
        config=config,
    )

    # Compile with multi-task losses
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )

    losses = {
        "colors": tf.keras.losses.BinaryCrossentropy(name="colors_loss"),
        "keywords": tf.keras.losses.BinaryCrossentropy(name="keywords_loss"),
        "leaders": tf.keras.losses.BinaryCrossentropy(name="leaders_loss"),
        "strategies": tf.keras.losses.BinaryCrossentropy(name="strategies_loss"),
    }

    loss_weights = {
        "colors": 1.0,
        "keywords": 1.0,
        "leaders": 2.0,  # Higher weight for leaders (more important)
        "strategies": 0.5,
    }

    metrics = {
        "colors": [tf.keras.metrics.BinaryAccuracy(name="colors_acc"), tf.keras.metrics.Precision(name="colors_prec"), tf.keras.metrics.Recall(name="colors_rec")],
        "keywords": [tf.keras.metrics.BinaryAccuracy(name="keywords_acc")],
        "leaders": [tf.keras.metrics.BinaryAccuracy(name="leaders_acc"), tf.keras.metrics.Precision(name="leaders_prec"), tf.keras.metrics.Recall(name="leaders_rec")],
        "strategies": [tf.keras.metrics.BinaryAccuracy(name="strategies_acc")],
    }

    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

    # Callbacks
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "intent_classifier.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    if early_stopping_patience:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save vocabularies
    vocab_dir = output_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    # Save color vocabulary
    with (vocab_dir / "color_vocabulary.json").open("w") as f:
        json.dump({"color_to_index": color_to_index, "index_to_color": index_to_color}, f, indent=2)

    # Save keyword vocabulary
    with (vocab_dir / "keyword_vocabulary.json").open("w") as f:
        json.dump({"keyword_to_index": keyword_to_index, "index_to_keyword": index_to_keyword}, f, indent=2)

    # Save leader vocabulary
    with (vocab_dir / "leader_vocabulary.json").open("w") as f:
        json.dump({"leader_to_index": leader_to_index, "index_to_leader": index_to_leader}, f, indent=2)

    # Save strategy vocabulary
    with (vocab_dir / "strategy_vocabulary.json").open("w") as f:
        json.dump({"strategy_to_index": strategy_to_index, "index_to_strategy": index_to_strategy}, f, indent=2)

    # Save leader IDs list
    with (vocab_dir / "leader_ids.json").open("w") as f:
        json.dump(leader_ids, f, indent=2)

    # Save prompt vectorizer vocabulary
    prompt_vocab_path = vocab_dir / "prompt_vocabulary.txt"
    with prompt_vocab_path.open("w") as f:
        for token in prompt_vectorizer.get_vocabulary():
            f.write(f"{token}\n")

    print(f"\nModel saved to: {checkpoint_path}")
    print(f"Vocabularies saved to: {vocab_dir}")


def main() -> None:
    # Configure TensorFlow and system for optimal CPU performance
    configure_tensorflow_for_cpu()
    configure_process_priority()
    
    parser = argparse.ArgumentParser(description="Train intent classifier model")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to intent examples JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for model and vocabularies")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--keyword-vocab-size", type=int, default=1000, help="Keyword vocabulary size")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Data root directory")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension",
    )
    parser.add_argument("--num-encoder-layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff-dim", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")

    args = parser.parse_args()

    config = IntentClassifierConfig(
        embedding_dim=args.embedding_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout_rate=args.dropout_rate,
    )

    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        config=config,
        early_stopping_patience=args.early_stopping_patience,
        keyword_vocab_size=args.keyword_vocab_size,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()

