"""
Utilities for loading and formatting game rules for ML training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RULES_SUMMARY = """One Piece TCG Rules: Decks must have exactly 50 cards (excluding Leader). Max 4 copies per card. Cards must match Leader color identity. Players start with 1 Leader (5000 power) and 5 Life cards. Don!! cards: gain 1 per turn (max 10 on field), attach for +1000 power or use for costs. Characters can be Active or Rested. To attack: rest Character/Leader and declare target. When attacked: use Counter cards or activate [Blocker] Characters. Life cards revealed on damage; add 1 to hand when revealed. Win by reducing opponent Life to 0. Card types: Leader, Character, Event, Stage. Colors: Red, Green, Blue, Purple, Yellow, Black. Key mechanics: [Blocker], [Rush], [Double Attack], [Trigger], [On Play], [When Attacking]. [Once Per Turn] abilities usable once per turn. [Activate: Main] abilities usable during main phase."""


def load_rules_summary(data_root: Optional[Path] = None) -> str:
    """
    Load a condensed rules summary from the extracted rules JSON.
    Falls back to a hardcoded summary if the file is not found.
    
    Args:
        data_root: Root directory containing data/manual/rules_extracted.json
        
    Returns:
        Condensed rules summary string suitable for prepending to training prompts.
    """
    if data_root is None:
        data_root = Path("data")
    
    rules_path = data_root / "manual" / "rules_extracted.json"
    
    if not rules_path.exists():
        logger.warning("Rules file not found at %s, using fallback summary", rules_path)
        return RULES_SUMMARY
    
    try:
        with rules_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        
        # Extract key rules from the structured data
        summary_parts = []
        
        # Get summary if available
        if "summary" in payload and payload["summary"]:
            raw_summary = payload["summary"]
            # Take first 500 characters of summary as condensed version
            summary_parts.append(raw_summary[:500].strip())
        
        # Extract key sections from rule_manual (prioritized)
        if "sources" in payload and "rule_manual" in payload["sources"]:
            rule_manual = payload["sources"]["rule_manual"]
            if "sections" in rule_manual:
                # Look for key sections about deck building and game rules
                key_section_titles = ["Decks", "Setting Up", "Game Flow", "Victory", "Areas"]
                for section in rule_manual["sections"][:10]:  # Limit to first 10 sections
                    title = section.get("title", "").lower()
                    if any(keyword in title for keyword in key_section_titles):
                        paragraphs = section.get("paragraphs", [])
                        if paragraphs:
                            # Take first 2 paragraphs of key sections
                            summary_parts.append(f"{section.get('title', '')}: {' '.join(paragraphs[:2])}")
        
        if summary_parts:
            # Combine and truncate to reasonable length for training
            combined = " ".join(summary_parts)
            # Limit to ~400 characters to keep prompts manageable
            if len(combined) > 400:
                combined = combined[:400].rsplit(" ", 1)[0] + "..."
            return f"One Piece TCG Rules: {combined}"
        
        return RULES_SUMMARY
        
    except Exception as exc:
        logger.warning("Failed to load rules from %s: %s, using fallback", rules_path, exc)
        return RULES_SUMMARY


def augment_prompt_with_rules(prompt: str, rules_summary: str, probability: float = 0.3) -> str:
    """
    Optionally prepend rules summary to a prompt.
    
    Args:
        prompt: Original training prompt
        rules_summary: Condensed rules summary
        probability: Probability (0.0-1.0) of augmenting the prompt with rules
        
    Returns:
        Augmented prompt if random draw succeeds, otherwise original prompt
    """
    import random
    
    if random.random() < probability:
        return f"{rules_summary}\n\n{prompt}"
    return prompt

