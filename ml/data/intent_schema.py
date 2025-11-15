"""
Typed schema definitions for intent classification training examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class IntentExample:
    """
    Training example for intent classification model.
    
    Links a natural-language prompt to structured intent labels:
    - Colors: which colors are mentioned/requested
    - Keywords: relevant keywords extracted from prompt
    - Leader IDs: which leaders match the prompt
    - Strategy tags: inferred strategy types (rush, control, etc.)
    """

    prompt: str
    colors: List[str]
    keywords: List[str]
    leader_ids: List[str]
    strategy_tags: List[str]
    split: Optional[str] = None

    def to_record(self) -> Dict[str, object]:
        """
        Serialise the example into a JSON-friendly dictionary.
        """
        return {
            "prompt": self.prompt,
            "colors": self.colors,
            "keywords": self.keywords,
            "leader_ids": self.leader_ids,
            "strategy_tags": self.strategy_tags,
            "split": self.split,
        }

    @classmethod
    def from_record(cls, payload: Dict[str, object]) -> "IntentExample":
        """
        Deserialise from a JSON-friendly dictionary.
        """
        return cls(
            prompt=str(payload.get("prompt", "")),
            colors=list(payload.get("colors") or []),
            keywords=list(payload.get("keywords") or []),
            leader_ids=list(payload.get("leader_ids") or []),
            strategy_tags=list(payload.get("strategy_tags") or []),
            split=payload.get("split"),
        )

