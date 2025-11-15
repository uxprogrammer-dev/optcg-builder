"""
Centralised schema definitions for decks, prompts, and training splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class DeckConfig:
    language: str = "en"
    main_deck_size: int = 50
    max_total_cards: int = 51  # leader + 50 main deck
    pad_token: str = "<PAD>"
    start_token: str = "<BOS>"
    end_token: str = "<EOS>"
    unknown_token: str = "<UNK>"


@dataclass(frozen=True)
class PromptConfig:
    max_length: int = 256
    vocab_size: int = 16000
    lowercase: bool = True
    reserved_tokens: Tuple[str, ...] = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
    paraphrase_dropout: float = 0.1
    rules_augmentation_rate: float = 0.3  # Probability of prepending rules to prompts


@dataclass
class TrainingSplits:
    proportions: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )

    def normalised(self) -> Dict[str, float]:
        total = sum(self.proportions.values())
        if total == 0:
            raise ValueError("Training split proportions must sum to > 0.")
        return {name: value / total for name, value in self.proportions.items()}

