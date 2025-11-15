"""
Typed schema definitions for cards, decks, and prompt/deck training examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class CardRecord:
    """
    Normalised representation of a single card entry.

    The raw JSON files contain nested objects for attributes, images, notes, etc.
    We flatten the pieces that are relevant to model training while preserving
    a copy of the untouched payload for forward compatibility.
    """

    id: str
    code: str
    name: str
    type: str
    color: Optional[str]
    rarity: Optional[str]
    cost: Optional[int]
    power: Optional[int]
    counter: Optional[str]
    attribute: Optional[str]
    family: Optional[str]
    ability: Optional[str]
    trigger: Optional[str]
    set_name: Optional[str]
    raw: Dict[str, object]

    @property
    def is_leader(self) -> bool:
        return self.type.upper() == "LEADER"

    @property
    def canonical_id(self) -> str:
        """
        Returns the canonical identifier used when constructing deck lists.

        Some card printings share the same code but differ by suffix (e.g. _p1).
        During deck construction we can choose to keep the fully-qualified id or
        fall back to the shared code depending on the experiment.
        """
        return self.id


@dataclass
class DeckSchema:
    """
    Representation of a deck suitable for model training and evaluation.

    The OPTCG game uses a single leader card and a 50-card main deck with optional
    don cards that are implicit in the format. We model a deck as a fixed-length
    sequence of card identifiers so that sequence-to-sequence models can emit the
    deck one card at a time.
    """

    leader_id: str
    main_deck: List[str]
    sideboard: Optional[List[str]] = None
    archetype: Optional[str] = None
    source_cards: Optional[List[str]] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_sequence(self, pad_to: Optional[int] = None, pad_token: str = "<PAD>") -> List[str]:
        """
        Flatten the deck into a single sequence:
        [leader_id, *main_deck, *sideboard?]
        Optionally pad / truncate to a fixed length.
        """
        sequence: List[str] = [self.leader_id, *self.main_deck]
        if self.sideboard:
            sequence.extend(self.sideboard)
        if pad_to is None:
            return sequence
        cropped = sequence[:pad_to]
        if len(cropped) < pad_to:
            cropped.extend([pad_token] * (pad_to - len(cropped)))
        return cropped

    @property
    def deck_size(self) -> int:
        return len(self.main_deck)

    def validate(
        self,
        allowed_main_deck_size: Sequence[int] = (50,),
        allow_duplicates: bool = True,
    ) -> None:
        """
        Basic structural validation to ensure the schema reflects game rules.
        Raises AssertionError on invalid decks.
        """
        assert self.leader_id, "Deck must include a leader id."
        if allowed_main_deck_size:
            assert self.deck_size in allowed_main_deck_size, (
                f"Expected main deck size in {allowed_main_deck_size}, "
                f"got {self.deck_size}."
            )
        if not allow_duplicates:
            seen = set()
            for card_id in self.main_deck:
                msg = f"Duplicate card {card_id} not allowed."
                assert card_id not in seen, msg
                seen.add(card_id)


@dataclass
class PromptDeckExample:
    """
    Training pair linking a natural-language prompt to a deck schema.
    """

    prompt: str
    deck: DeckSchema
    prompt_style: Optional[str] = None
    quality_tags: Optional[List[str]] = None
    split: Optional[str] = None
    leader_ability: Optional[str] = None
    leader_subtypes: Optional[List[str]] = None
    leader_colors: Optional[List[str]] = None
    card_synergies: Optional[Dict[str, List[str]]] = None
    card_rationales: Optional[Dict[str, str]] = None

    def to_record(self) -> Dict[str, object]:
        """
        Serialise the example into a JSON-friendly dictionary.
        """
        deck_payload = {
            "leader_id": self.deck.leader_id,
            "main_deck": self.deck.main_deck,
            "sideboard": self.deck.sideboard,
            "archetype": self.deck.archetype,
            "metadata": self.deck.metadata,
        }
        return {
            "prompt": self.prompt,
            "prompt_style": self.prompt_style,
            "quality_tags": self.quality_tags,
            "split": self.split,
            "deck": deck_payload,
            "leader_ability": self.leader_ability,
            "leader_subtypes": self.leader_subtypes,
            "leader_colors": self.leader_colors,
            "card_synergies": self.card_synergies,
            "card_rationales": self.card_rationales,
        }

    @classmethod
    def from_record(cls, payload: Dict[str, object]) -> "PromptDeckExample":
        deck_payload = payload.get("deck", {})
        deck = DeckSchema(
            leader_id=deck_payload.get("leader_id"),
            main_deck=list(deck_payload.get("main_deck") or []),
            sideboard=deck_payload.get("sideboard"),
            archetype=deck_payload.get("archetype"),
            metadata=deck_payload.get("metadata") or {},
        )
        return cls(
            prompt=str(payload.get("prompt", "")),
            deck=deck,
            prompt_style=payload.get("prompt_style"),
            quality_tags=payload.get("quality_tags"),
            split=payload.get("split"),
            leader_ability=payload.get("leader_ability"),
            leader_subtypes=payload.get("leader_subtypes"),
            leader_colors=payload.get("leader_colors"),
            card_synergies=payload.get("card_synergies"),
            card_rationales=payload.get("card_rationales"),
        )

