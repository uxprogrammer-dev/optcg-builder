#!/usr/bin/env python3
"""
Test script to verify training loop works with copy count labels and updated weights.

This script runs training on a small subset (200 decks) for a few epochs to ensure
the training loop works smoothly before training on the full dataset.
"""

from pathlib import Path
import sys

# Add parent directory to path to import ml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.datasets import load_examples_jsonl, write_examples_jsonl
from ml.training.train import train


def create_test_dataset(input_path: Path, output_path: Path, num_decks: int = 200) -> None:
    """
    Create a small test dataset by taking the first N decks from the input dataset.
    
    Args:
        input_path: Path to the full dataset JSONL file
        output_path: Path where the test dataset will be written
        num_decks: Number of decks to include in the test dataset
    """
    print(f"Loading dataset from {input_path}...")
    examples = load_examples_jsonl(input_path)
    print(f"Loaded {len(examples)} examples")
    
    # Take first num_decks examples
    test_examples = examples[:num_decks]
    print(f"Creating test dataset with {len(test_examples)} examples")
    
    # Ensure all examples are marked as "train" split for simplicity
    for example in test_examples:
        example.split = "train"
    
    # Write test dataset
    write_examples_jsonl(test_examples, output_path)
    print(f"Test dataset written to {output_path}")


def main():
    """Run a small training test."""
    # Default paths
    default_input = Path("ml/artifacts/synthetic_prompt_deck.jsonl")
    default_output = Path("ml/artifacts/test_prompt_deck_200.jsonl")
    default_output_dir = Path("models/test_run")
    
    # Check if input file exists
    if not default_input.exists():
        print(f"Error: Input dataset not found at {default_input}")
        print("Please provide a valid dataset path.")
        sys.exit(1)
    
    # Create test dataset
    print("=" * 60)
    print("Creating test dataset (200 decks)...")
    print("=" * 60)
    create_test_dataset(default_input, default_output, num_decks=200)
    
    # Run training with updated weights
    print("\n" + "=" * 60)
    print("Running training test (3 epochs)...")
    print("=" * 60)
    print("Configuration:")
    print(f"  - Dataset: {default_output}")
    print(f"  - Epochs: 3")
    print(f"  - Batch size: 16 (smaller for test)")
    print(f"  - freq_hist_weight: 300.0 (increased from 200.0)")
    print(f"  - sequence_level_weight: 25.0 (reduced from 400.0)")
    print(f"  - Label generation: Copy count / 4 (already implemented)")
    print(f"  - Decoder: Samples with replacement up to 4 copies (already implemented)")
    print()
    
    train(
        dataset_path=default_output,
        output_dir=default_output_dir,
        epochs=3,
        batch_size=16,  # Smaller batch size for test
        learning_rate=3e-4,
        seed=1234,
        freq_hist_weight=300.0,  # Updated weight
        sequence_level_weight=25.0,  # Updated weight (20-30 range)
        save_checkpoints=True,
        early_stopping_patience=None,  # Disable early stopping for short test
    )
    
    print("\n" + "=" * 60)
    print("Training test completed successfully!")
    print("=" * 60)
    print(f"Model saved to: {default_output_dir}")
    print("\nNext steps:")
    print("  1. Review the training logs to verify the training loop works")
    print("  2. Check that freq_hist and sequence_level losses are being computed")
    print("  3. If everything looks good, run full training with:")
    print(f"     python -m ml.training.train --dataset {default_input} --epochs 20")


if __name__ == "__main__":
    main()
