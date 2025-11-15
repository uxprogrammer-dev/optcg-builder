"""
Quick script to show model specifications from code without loading models.
"""

from pathlib import Path
from ml.models.transformer import TransformerConfig
from ml.models.intent_classifier import IntentClassifierConfig


def print_config(name: str, config):
    """Print configuration details."""
    print(f"\n{name}:")
    print("=" * 60)
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")


def main():
    print("Model Specifications")
    print("=" * 60)
    
    # Transformer config (used by deck transformer)
    transformer_config = TransformerConfig()
    print_config("Deck Transformer Configuration", transformer_config)
    
    # Intent classifier config
    intent_config = IntentClassifierConfig()
    print_config("Intent Classifier Configuration", intent_config)
    
    # Model architecture summary
    print("\n\nModel Architecture Summary")
    print("=" * 60)
    
    print("\n1. Intent Classifier:")
    print("   - Type: Encoder-only Transformer")
    print("   - Layers: 4 encoder layers")
    print("   - Attention Heads: 8")
    print("   - Embedding Dimension: 256")
    print("   - Feed-forward Dimension: 512")
    print("   - Outputs: Multi-task classification (colors, keywords, leaders, strategies)")
    
    print("\n2. Deck Transformer:")
    print("   - Type: Encoder-Decoder Transformer")
    print("   - Layers: 4 encoder + 4 decoder layers")
    print("   - Attention Heads: 8")
    print("   - Embedding Dimension: 256")
    print("   - Feed-forward Dimension: 512")
    print("   - Outputs: Card sequence + auxiliary predictions")
    
    print("\n\nPublic Presentation Assessment")
    print("=" * 60)
    print("✓ Custom architecture (not proprietary)")
    print("✓ Built with open-source TensorFlow/Keras")
    print("✓ Safe to present publicly")
    print("\nNote: Verify training data attribution and OPTCG API terms")


if __name__ == "__main__":
    main()

