"""
Model architectures for prompt→deck generation.
"""

from .transformer import TransformerConfig, build_deck_transformer

__all__ = ["TransformerConfig", "build_deck_transformer"]

