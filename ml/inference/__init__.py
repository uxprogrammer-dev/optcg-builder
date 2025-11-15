"""
Inference helpers for turning prompts into deck predictions.
"""

from .generate import beam_search_generate, greedy_generate

__all__ = ["beam_search_generate", "greedy_generate"]

