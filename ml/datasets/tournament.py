"""
Utilities for converting real-world tournament decks into training examples.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from ..data.card_schema import DeckSchema, PromptDeckExample
from ..data.loader import CardRecord, CardRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TournamentDeck:
    """
    Normalised representation of a tournament deck entry.
    """

    name: str
    leader_id: str
    leader: CardRecord
    main_deck: List[str]
    metadata: Dict[str, object]
    prompt: str
    placement: Optional[str]


def load_tournament_examples(
    data_root: Path,
    tournaments_dir: Optional[Path] = None,
    language: str = "en",
    repository: Optional[CardRepository] = None,
) -> List[PromptDeckExample]:
    """
    Load tournament deck JSON files and convert them into PromptDeckExample objects.
    """

    card_repository = repository or CardRepository(data_root=data_root, language=language)
    card_repository.load()

    directory = tournaments_dir or (Path(data_root) / "tournaments")
    if not directory.exists():
        logger.warning("Tournament directory not found: %s", directory)
        return []

    examples: List[PromptDeckExample] = []
    for json_path in sorted(directory.glob("*.json")):
        payload = _read_payload(json_path)
        if payload is None:
            continue

        for index, entry in enumerate(payload):
            try:
                tournament_deck = _normalise_entry(entry, card_repository, json_path)
            except ValueError as error:
                # Skip decks without leaders or other validation errors
                error_msg = str(error)
                if "could not identify leader card" in error_msg.lower():
                    logger.debug("Skipping deck without leader at %s[%s]: %s", json_path.name, index, error_msg)
                else:
                    logger.warning("Skipping tournament deck at %s[%s]: %s", json_path.name, index, error_msg)
                continue

            example = PromptDeckExample(
                prompt=tournament_deck.prompt,
                deck=DeckSchema(
                    leader_id=tournament_deck.leader_id,
                    main_deck=tournament_deck.main_deck,
                    archetype=tournament_deck.name,
                    metadata=tournament_deck.metadata,
                ),
                prompt_style="tournament",
                quality_tags=_quality_tags(tournament_deck.placement),
                split="train",
                leader_ability=tournament_deck.leader.ability,
                leader_subtypes=_extract_leader_subtypes(tournament_deck.leader),
                leader_colors=_extract_leader_colors(tournament_deck.leader),
                card_synergies=extract_card_synergies(tournament_deck),
                card_rationales=_generate_card_rationales(tournament_deck, card_repository) or None,
            )
            examples.append(example)

    return examples


def _read_payload(path: Path) -> Optional[List[Mapping[str, object]]]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except json.JSONDecodeError as error:
        logger.warning("Failed to parse %s: %s", path, error)
        return None

    if not isinstance(payload, list):
        logger.warning("Tournament file %s is not a list of deck entries.", path)
        return None
    return [entry for entry in payload if isinstance(entry, Mapping)]


def _normalise_entry(
    entry: Mapping[str, object],
    repository: CardRepository,
    source_path: Path,
) -> TournamentDeck:
    decklist = entry.get("decklist")
    if not isinstance(decklist, Mapping):
        raise ValueError("entry missing 'decklist' mapping")
    
    # Early validation: ensure decklist is not empty
    if not decklist:
        deck_name = entry.get("deck_name") or "unknown"
        raise ValueError(f"decklist is empty for deck '{deck_name}'")

    leader_record: Optional[CardRecord] = None
    leader_id: Optional[str] = None
    main_deck: List[str] = []
    missing_cards: List[str] = []

    for raw_card_id, raw_quantity in decklist.items():
        if raw_card_id is None:
            continue

        resolved_quantity = _coerce_quantity(raw_quantity)
        if resolved_quantity <= 0:
            continue

        resolved_id, card_record = _resolve_card_id(str(raw_card_id), repository)
        if card_record is None:
            missing_cards.append(str(raw_card_id))
            continue

        if card_record.is_leader:
            leader_id = resolved_id
            leader_record = card_record
            if resolved_quantity != 1:
                logger.debug(
                    "Leader %s from %s has quantity %s – expected 1.",
                    resolved_id,
                    source_path.name,
                    resolved_quantity,
                )
            continue

        main_deck.extend([resolved_id] * resolved_quantity)

    # Explicitly check and reject decks without leaders
    if leader_id is None or leader_record is None:
        deck_name = entry.get("deck_name") or "unknown"
        logger.warning(
            "Skipping tournament deck '%s' from %s: no leader card found in decklist.",
            deck_name,
            source_path.name,
        )
        raise ValueError(f"could not identify leader card for deck '{deck_name}'")

    if missing_cards:
        logger.debug(
            "Tournament deck '%s' from %s skipped %s unknown cards.",
            entry.get("deck_name") or leader_record.name,
            source_path.name,
            ", ".join(missing_cards),
        )

    # Normalize deck to exactly 50 cards for training consistency
    main_deck = _normalize_deck_size(
        main_deck=main_deck,
        leader_id=leader_id,
        leader_record=leader_record,
        repository=repository,
        target_size=50,
        source_path=source_path,
    )

    if len(main_deck) != 50:
        logger.warning(
            "Tournament deck from %s normalized to %s cards (leader=%s). Original size may have differed.",
            source_path.name,
            len(main_deck),
            leader_id,
        )

    metadata = _build_metadata(entry, source_path, leader_record)
    prompt = _build_prompt(entry.get("deck_name"), leader_record)

    tournament_deck = TournamentDeck(
        name=str(entry.get("deck_name") or leader_record.name),
        leader_id=leader_id,
        leader=leader_record,
        main_deck=main_deck,
        metadata=metadata,
        prompt=prompt,
        placement=entry.get("placement"),
    )

    # Validate structure - decks should now be exactly 50 cards after normalization
    DeckSchema(leader_id=leader_id, main_deck=main_deck).validate(allowed_main_deck_size=(50,))

    return tournament_deck


def _normalize_deck_size(
    main_deck: List[str],
    leader_id: str,
    leader_record: CardRecord,
    repository: CardRepository,
    target_size: int = 50,
    source_path: Optional[Path] = None,
) -> List[str]:
    """
    Normalize deck to exactly target_size cards.
    
    If deck has fewer cards, fills with appropriate cards matching leader colors/types.
    If deck has more cards, trims to target_size while preserving important cards.
    """
    current_size = len(main_deck)
    
    if current_size == target_size:
        return main_deck
    
    # Get leader colors and subtypes for filling
    leader_colors = _extract_leader_colors(leader_record)
    leader_subtypes = _extract_leader_subtypes(leader_record)
    
    if current_size < target_size:
        # Need to add cards
        needed = target_size - current_size
        counts = Counter(main_deck)
        added = []
        
        # Find candidate cards matching leader colors/types
        candidates: List[Tuple[int, str, CardRecord]] = []
        for card_id, card_record in repository.cards.items():
            if card_id == leader_id or card_record.is_leader:
                continue
            if counts[card_id] >= 4:  # Max copies
                continue
            
            # Prefer cards matching leader colors
            card_colors = _extract_leader_colors(card_record)
            color_match = not leader_colors or any(c in leader_colors for c in card_colors)
            
            # Prefer cards matching leader subtypes
            card_subtypes = _extract_leader_subtypes(card_record)
            subtype_match = not leader_subtypes or any(s in leader_subtypes for s in card_subtypes)
            
            # Score: color match = 2, subtype match = 1, neither = 0
            score = (2 if color_match else 0) + (1 if subtype_match else 0)
            candidates.append((score, card_id, card_record))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Add cards up to target size
        for score, card_id, _ in candidates:
            if len(added) >= needed:
                break
            if counts[card_id] < 4:
                added.append(card_id)
                counts[card_id] += 1
        
        # If still short, repeat existing cards (up to 4 copies)
        if len(added) < needed:
            existing_cards = list(main_deck)
            idx = 0
            while len(added) < needed and existing_cards:
                candidate = existing_cards[idx % len(existing_cards)]
                if counts[candidate] < 4:
                    added.append(candidate)
                    counts[candidate] += 1
                idx += 1
                # Safety: prevent infinite loop if all cards are at max copies
                if idx >= len(existing_cards) * 4:
                    # Force add cards even if at max copies (shouldn't happen, but safety)
                    break
        
        result = main_deck + added
        # Ensure exactly target_size cards
        result = result[:target_size]
        
        # Final safety: if still short, pad with last card (shouldn't happen)
        while len(result) < target_size and result:
            result.append(result[-1])
        
        logger.debug(
            "Normalized deck from %s cards to %s cards (added %s cards) for leader %s",
            current_size,
            len(result),
            len(added),
            leader_id,
        )
        return result[:target_size]
    
    else:
        # Need to remove cards (keep most important ones)
        # Simple approach: keep first target_size cards
        # In a more sophisticated version, we could score cards by importance
        logger.debug(
            "Normalized deck from %s cards to %s cards (removed %s cards) for leader %s",
            current_size,
            target_size,
            current_size - target_size,
            leader_id,
        )
        return main_deck[:target_size]


def _coerce_quantity(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _resolve_card_id(card_id: str, repository: CardRepository) -> Tuple[str, Optional[CardRecord]]:
    normalised = card_id.strip().upper()

    record = repository.by_id(normalised)
    if record:
        return record.id, record

    base_id = normalised.split("_")[0]
    if base_id != normalised:
        base_record = repository.by_id(base_id)
        if base_record:
            return base_record.id, base_record

    candidates = repository.by_code(base_id)
    if candidates:
        return candidates[0].id, candidates[0]

    return normalised, None


def _build_prompt(deck_name: Optional[str], leader: CardRecord) -> str:
    leader_name = leader.name
    if deck_name:
        deck_name_clean = deck_name.strip()
        if deck_name_clean.lower() in leader_name.lower():
            return f"Create a competitive deck for {leader_name}."
        return f"Create a competitive {deck_name_clean} deck led by {leader_name}."
    return f"Create a competitive deck for {leader_name}."


def _build_metadata(
    entry: Mapping[str, object],
    source_path: Path,
    leader: CardRecord,
) -> Dict[str, object]:
    metadata_keys = ("deck_name", "author", "date", "country", "tournament", "placement", "host")
    metadata: Dict[str, object] = {
        "source": "tournament",
        "source_file": source_path.name,
        "leader_name": leader.name,
    }

    for key in metadata_keys:
        value = entry.get(key)
        if value:
            metadata[key] = value

    return metadata


def extract_card_synergies(deck: TournamentDeck, max_links: int = 8) -> Dict[str, List[str]]:
    """
    Extract simple co-occurrence synergies from a tournament deck.
    """

    if not deck.main_deck:
        return {}

    co_occurrence: Dict[str, Counter[str]] = defaultdict(Counter)
    cards = [card_id.upper() for card_id in deck.main_deck]

    for index, card_id in enumerate(cards):
        for other in cards[index + 1 :]:
            if other == card_id:
                continue
            co_occurrence[card_id][other] += 1
            co_occurrence[other][card_id] += 1

    result: Dict[str, List[str]] = {}
    for card_id, counter in co_occurrence.items():
        if not counter:
            continue
        ranked = [other for other, _ in counter.most_common(max_links)]
        if ranked:
            result[card_id] = ranked
    return result


def _extract_leader_subtypes(leader: CardRecord) -> List[str]:
    family = (leader.family or "").strip()
    subtypes = [part.strip() for part in family.replace("&", "/").split("/") if part.strip()]
    ordered: List[str] = []
    seen: Dict[str, None] = {}
    for subtype in subtypes:
        key = subtype.lower()
        if key not in seen:
            seen[key] = None
            ordered.append(subtype)
    return ordered


def _extract_leader_colors(leader: CardRecord) -> List[str]:
    color_field = (leader.color or "").strip()
    colors = [part.strip() for part in color_field.replace("&", "/").split("/") if part.strip()]
    ordered: List[str] = []
    seen: Dict[str, None] = {}
    for color in colors:
        key = color.lower()
        if key not in seen:
            seen[key] = None
            ordered.append(color)
    return ordered


def _ability_keywords(text: Optional[str]) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    keywords = [
        "draw",
        "search",
        "don",
        "counter",
        "blocker",
        "rush",
        "double attack",
        "k.o.",
        "remove",
        "rest",
        "trash",
        "life",
    ]
    return [keyword for keyword in keywords if keyword in lowered]


def _generate_card_rationales(
    deck: TournamentDeck,
    repository: CardRepository,
    max_synergy_keywords: int = 3,
) -> Dict[str, str]:
    leader = deck.leader
    leader_subtypes = _extract_leader_subtypes(leader)
    leader_colors = _extract_leader_colors(leader)
    leader_keywords = set(_ability_keywords(leader.ability))

    rationales: Dict[str, str] = {}
    for card_id in deck.main_deck:
        if card_id in rationales:
            continue
        record = repository.by_id(card_id)
        if not record:
            continue

        reasons: List[str] = []

        card_subtypes = [subtype for subtype in _extract_leader_subtypes(record) if subtype in leader_subtypes]
        if card_subtypes:
            reasons.append(
                f"Shares {', '.join(card_subtypes)} subtype(s) with {leader.name} for archetype consistency.",
            )

        card_colors = _extract_leader_colors(record)
        if leader_colors and card_colors and set(card_colors).issubset(set(leader_colors)):
            reasons.append("Matches leader color identity to keep the deck consistent.")

        card_keywords = set(_ability_keywords(record.ability))
        keyword_overlap = sorted(card_keywords & leader_keywords)[:max_synergy_keywords]
        if keyword_overlap:
            reasons.append(f"Aligns with the leader ability focus via {', '.join(keyword_overlap)} effects.")
        elif card_keywords:
            reasons.append(
                f"Adds {', '.join(sorted(card_keywords)[:max_synergy_keywords])} utility to cover matchup needs.",
            )

        if not reasons:
            role = (record.type or "card").lower()
            reasons.append(f"Reliable {role} option that complements {leader.name}'s game plan.")

        rationales[card_id] = " ".join(reasons)

    return rationales


def _quality_tags(placement: Optional[str]) -> List[str]:
    tags = ["tournament", "meta"]
    if not placement:
        return tags

    placement_lower = placement.lower()
    placement_upper = placement.upper()

    if "1st" in placement_lower or "champion" in placement_lower or "winner" in placement_lower:
        tags.append("winner")
    elif any(term in placement_lower for term in ("2nd", "3rd", "runner", "finalist")):
        tags.append("top-3")
    elif any(term in placement_upper for term in ("T4", "T8", "T16", "TOP4", "TOP8", "TOP16")):
        tags.append("top-16")
    return tags


__all__ = ["load_tournament_examples", "TournamentDeck", "extract_card_synergies"]


