"""
Utilities for loading and generating intent classification training examples.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set

from ..data import PromptDeckExample
from ..data.intent_schema import IntentExample

logger = logging.getLogger(__name__)

# Strategy keywords mapping
STRATEGY_KEYWORDS = {
    "rush": ["rush", "aggressive", "fast", "quick", "early", "pressure"],
    "control": ["control", "defensive", "slow", "late", "grind"],
    "combo": ["combo", "synergy", "interaction", "chain"],
    "midrange": ["midrange", "balanced", "versatile"],
    "aggro": ["aggro", "aggressive", "rush", "fast"],
}

# Color names (standardized)
COLORS = ["red", "blue", "green", "yellow", "purple", "black"]


def load_intent_examples(path: Path) -> List[IntentExample]:
    """
    Load intent examples from a JSONL file.
    """
    examples: List[IntentExample] = []
    if not path.exists():
        logger.warning("Intent examples file not found: %s", path)
        return examples

    with path.open("r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                example = IntentExample.from_record(payload)
                examples.append(example)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                logger.warning("Skipping invalid line %d in %s: %s", line_num, path, error)
                continue

    return examples


def save_intent_examples(examples: List[IntentExample], path: Path) -> None:
    """
    Save intent examples to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for example in examples:
            record = example.to_record()
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    return text.lower().replace(".", " ").replace("-", " ").strip()


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    normalized = _normalize_text(text)
    # Split on whitespace and filter empty strings
    tokens = [t.strip() for t in normalized.split() if t.strip()]
    return tokens


def _extract_keywords_from_prompt(
    prompt: str,
    leader_keywords: List[str],
    stopwords: Set[str] = None,
) -> List[str]:
    """
    Extract relevant keywords from prompt by matching against leader keywords.
    """
    if stopwords is None:
        stopwords = {
            "the",
            "of",
            "and",
            "or",
            "to",
            "in",
            "a",
            "an",
            "deck",
            "around",
            "strategy",
            "build",
            "create",
            "make",
            "for",
            "with",
            "that",
            "this",
            "is",
            "are",
            "was",
            "were",
        }

    prompt_tokens = set(_tokenize(prompt))
    matched_keywords: List[str] = []

    # Normalize leader keywords for matching
    normalized_leader_keywords = {}
    for kw in leader_keywords:
        normalized_kw = _normalize_text(kw)
        normalized_leader_keywords[normalized_kw] = kw

    # Check each prompt token against leader keywords
    for token in prompt_tokens:
        if token in stopwords or len(token) < 3:
            continue

        # Direct match
        if token in normalized_leader_keywords:
            matched_keywords.append(normalized_leader_keywords[token])
            continue

        # Substring match (token appears in keyword)
        for norm_kw, orig_kw in normalized_leader_keywords.items():
            if token in norm_kw or norm_kw in token:
                matched_keywords.append(orig_kw)
                break

    return list(set(matched_keywords))  # Deduplicate


def _extract_strategy_tags(prompt: str) -> List[str]:
    """
    Infer strategy tags from prompt keywords.
    """
    normalized_prompt = _normalize_text(prompt)
    prompt_tokens = set(_tokenize(normalized_prompt))

    matched_strategies: List[str] = []
    for strategy, keywords in STRATEGY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in prompt_tokens or keyword in normalized_prompt:
                matched_strategies.append(strategy)
                break

    return list(set(matched_strategies))


def _extract_colors_from_prompt(prompt: str) -> List[str]:
    """
    Extract color mentions from prompt.
    """
    normalized_prompt = _normalize_text(prompt)
    prompt_tokens = set(_tokenize(normalized_prompt))

    matched_colors: List[str] = []
    for color in COLORS:
        if color in prompt_tokens or color in normalized_prompt:
            matched_colors.append(color)

    return matched_colors


def generate_intent_from_deck_examples(
    examples: List[PromptDeckExample],
    leader_keywords_map: Dict[str, List[str]] = None,
) -> List[IntentExample]:
    """
    Convert PromptDeckExample instances to IntentExample instances.
    
    Args:
        examples: List of PromptDeckExample instances
        leader_keywords_map: Optional mapping of leader_id -> list of keywords
                            If not provided, keywords will be extracted from prompt only
    
    Returns:
        List of IntentExample instances
    """
    intent_examples: List[IntentExample] = []

    for example in examples:
        # Extract colors from leader_colors field
        colors = []
        if example.leader_colors:
            colors = [c.lower().strip() for c in example.leader_colors if c]

        # Also extract colors from prompt
        prompt_colors = _extract_colors_from_prompt(example.prompt)
        colors.extend(prompt_colors)
        colors = list(set(colors))  # Deduplicate

        # Extract keywords
        keywords = []
        if leader_keywords_map and example.deck.leader_id:
            leader_kws = leader_keywords_map.get(example.deck.leader_id, [])
            keywords = _extract_keywords_from_prompt(example.prompt, leader_kws)
        else:
            # Fallback: extract from prompt tokens (simpler)
            prompt_tokens = _tokenize(example.prompt)
            stopwords = {
                "the", "of", "and", "or", "to", "in", "a", "an", "deck",
                "around", "strategy", "build", "create", "make", "for",
                "with", "that", "this", "is", "are", "was", "were",
            }
            keywords = [t for t in prompt_tokens if t not in stopwords and len(t) >= 3]

        # Extract leader IDs
        leader_ids = []
        if example.deck.leader_id:
            leader_ids = [example.deck.leader_id]

        # Extract strategy tags
        strategy_tags = _extract_strategy_tags(example.prompt)

        intent_example = IntentExample(
            prompt=example.prompt,
            colors=colors,
            keywords=keywords,
            leader_ids=leader_ids,
            strategy_tags=strategy_tags,
            split=example.split or "train",
        )
        intent_examples.append(intent_example)

    return intent_examples

