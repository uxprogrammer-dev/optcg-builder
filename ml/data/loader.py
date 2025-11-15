"""
Utilities to load and normalise card data from the JSON exports.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .card_schema import CardRecord


class CardRepository:
    """
    Loads merged card data across sets and provides lookup helpers.

    Parameters
    ----------
    data_root:
        Path to the `data` directory containing `cards/<lang>/*.json`.
    language:
        Language code to scope the repo to. Defaults to \"en\".
    """

    def __init__(self, data_root: Path, language: str = "en") -> None:
        self.data_root = Path(data_root)
        self.language = language
        self._cards_by_id: Dict[str, CardRecord] = {}
        self._cards_by_code: Dict[str, List[CardRecord]] = {}

    # --------------------------------------------------------------------- #
    # Loading
    # --------------------------------------------------------------------- #
    def load(self, force: bool = False) -> None:
        if self._cards_by_id and not force:
            return

        cards_dir = self.data_root / "cards" / self.language
        if not cards_dir.exists():
            raise FileNotFoundError(f"Cards directory not found: {cards_dir}")

        loaded: Dict[str, CardRecord] = {}
        by_code: Dict[str, List[CardRecord]] = {}

        for json_path in sorted(cards_dir.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            for raw_card in payload:
                record = self._normalise_card(raw_card)
                loaded[record.id] = record
                by_code.setdefault(record.code, []).append(record)

        self._cards_by_id = loaded
        self._cards_by_code = by_code

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def cards(self) -> Mapping[str, CardRecord]:
        if not self._cards_by_id:
            self.load()
        return self._cards_by_id

    def by_id(self, card_id: str) -> Optional[CardRecord]:
        return self.cards.get(card_id)

    def by_code(self, code: str) -> List[CardRecord]:
        return list(self._cards_by_code.get(code, []))

    def leaders(self) -> Iterable[CardRecord]:
        for card in self.cards.values():
            if card.is_leader:
                yield card

    def archetype_groups(self) -> Dict[str, List[CardRecord]]:
        """
        Group cards by family/archetype string.
        """
        groups: Dict[str, List[CardRecord]] = {}
        for record in self.cards.values():
            if not record.family:
                continue
            groups.setdefault(record.family, []).append(record)
        return groups

    # ------------------------------------------------------------------ #
    # Export helpers
    # ------------------------------------------------------------------ #
    def as_dict(self) -> Dict[str, Dict[str, object]]:
        return {card_id: asdict(card) for card_id, card in self.cards.items()}

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _normalise_card(self, raw: Mapping[str, object]) -> CardRecord:
        # Defensive casting due to inconsistent typing in raw JSON.
        attribute_name = None
        if isinstance(raw.get("attribute"), dict):
            attribute_name = raw["attribute"].get("name")
        elif raw.get("attribute"):
            attribute_name = str(raw["attribute"])

        set_name = None
        if isinstance(raw.get("set"), dict):
            set_name = raw["set"].get("name")
        elif raw.get("set"):
            set_name = str(raw["set"])

        def _int_or_none(value: object) -> Optional[int]:
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return None

        return CardRecord(
            id=str(raw["id"]),
            code=str(raw.get("code", raw["id"])),
            name=str(raw.get("name")),
            type=str(raw.get("type")),
            color=str(raw.get("color")) if raw.get("color") else None,
            rarity=str(raw.get("rarity")) if raw.get("rarity") else None,
            cost=_int_or_none(raw.get("cost")),
            power=_int_or_none(raw.get("power")),
            counter=str(raw.get("counter")) if raw.get("counter") else None,
            attribute=attribute_name,
            family=str(raw.get("family")) if raw.get("family") else None,
            ability=str(raw.get("ability")) if raw.get("ability") else None,
            trigger=str(raw.get("trigger")) if raw.get("trigger") else None,
            set_name=set_name,
            raw=dict(raw),
        )

