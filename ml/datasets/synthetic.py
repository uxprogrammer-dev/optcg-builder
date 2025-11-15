"""
Synthetic prompt→deck dataset generation using heuristics over the card data.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..config import DeckConfig, PromptConfig, TrainingSplits
from ..data import CardRecord, CardRepository, DeckSchema, PromptDeckExample
from .tournament import load_tournament_examples

PROMPT_TEMPLATES: Tuple[str, ...] = (
    "Build a competitive {colors} deck led by {leader} that leverages {focus} synergies.",
    "Create a {colors} deck for {leader} emphasising {focus} tactics.",
    "Design a {colors} deck around {leader} with an emphasis on {focus}.",
    "Construct a {colors} strategy for {leader} that can counter {opposition}.",
)

OPPOSITION_TEMPLATES: Tuple[str, ...] = (
    "aggro threats",
    "control matchups",
    "midrange decks",
    "Don ramp strategies",
)

FOCUS_DEFAULTS: Tuple[str, ...] = (
    "balanced offense and defense",
    "board pressure",
    "card advantage",
    "Don acceleration",
    "rush mechanics",
)


def _normalise_color_label(color: Optional[str]) -> str:
    if not color:
        return "multicolor"
    return color.replace("/", " & ")


def _extract_focus_from_ability(ability: Optional[str]) -> Optional[str]:
    if not ability:
        return None
    ability_clean = re.sub(r"\[.*?\]", "", ability)  # remove bracketed cost text
    ability_clean = re.sub(r"\{.*?\}", "", ability_clean)
    ability_clean = ability_clean.replace("\n", " ").strip()
    if not ability_clean:
        return None
    sentences = re.split(r"[.!?]", ability_clean)
    for sentence in sentences:
        phrase = sentence.strip()
        if phrase:
            phrase = re.sub(r"\s+", " ", phrase.lower())
            words = phrase.split()
            if len(words) > 18:
                phrase = " ".join(words[:18])
            return phrase
    return None


@dataclass
class SyntheticDeckGenerator:
    repository: CardRepository
    deck_config: DeckConfig = DeckConfig()
    prompt_config: PromptConfig = PromptConfig()
    random_seed: int = 42
    max_duplicates_per_card: int = 4

    def __post_init__(self) -> None:
        self.rng = random.Random(self.random_seed)

    def generate_example(self, leader: CardRecord) -> PromptDeckExample:
        deck = self._build_deck(leader)
        prompt = self._build_prompt(leader, deck)
        leader_subtypes = self._split_multivalue(leader.family)
        leader_colors = self._split_multivalue(leader.color)
        leader_keywords = self._ability_keywords(leader.ability)

        card_rationales: Dict[str, str] = {}
        for card_id in deck.main_deck:
            if card_id in card_rationales:
                continue
            card_record = self.repository.by_id(card_id)
            if not card_record:
                continue
            card_rationales[card_id] = self._generate_card_rationale(
                card_record,
                leader,
                leader_subtypes,
                leader_colors,
                leader_keywords,
            )

        return PromptDeckExample(
            prompt=prompt,
            deck=deck,
            prompt_style="synthetic",
            leader_ability=leader.ability,
            leader_subtypes=leader_subtypes or None,
            leader_colors=leader_colors or None,
            card_synergies=self._build_synergy_map(deck.main_deck),
            card_rationales=card_rationales or None,
        )

    # ------------------------------------------------------------------ #
    def _build_deck(self, leader: CardRecord) -> DeckSchema:
        candidate_pool = self._candidate_pool(leader)
        main_deck = self._sample_main_deck(candidate_pool)
        deck = DeckSchema(
            leader_id=leader.canonical_id,
            main_deck=main_deck,
            archetype=self._infer_archetype(leader),
            metadata={
                "leader_name": leader.name,
                "leader_color": leader.color,
                "source": "synthetic",
            },
        )
        deck.validate(allowed_main_deck_size=(self.deck_config.main_deck_size,))
        return deck

    def _candidate_pool(self, leader: CardRecord) -> List[CardRecord]:
        colors = []
        if leader.color:
            colors = [color.strip() for color in leader.color.split("/")]
        family = leader.family

        pool: List[CardRecord] = []
        for card in self.repository.cards.values():
            if card.id == leader.id:
                continue
            if card.type.upper() == "LEADER":
                continue
            if colors and card.color:
                match = any(color in card.color for color in colors)
            else:
                match = True
            family_match = family and card.family and family in card.family
            if match:
                score = 2 if family_match else 1
                pool.extend([card] * score)
        if not pool:
            raise ValueError(f"No candidate cards found for leader {leader.id}.")
        return pool

    def _sample_main_deck(self, pool: Sequence[CardRecord]) -> List[str]:
        size = self.deck_config.main_deck_size
        selections: List[str] = []
        attempts = 0
        while len(selections) < size and attempts < size * 10:
            attempts += 1
            card = self.rng.choice(pool)
            count = selections.count(card.canonical_id)
            if count >= self.max_duplicates_per_card:
                continue
            selections.append(card.canonical_id)
        if len(selections) < size:
            raise RuntimeError("Failed to assemble deck with sufficient cards.")
        # Ensure exactly size cards (trim if somehow exceeded)
        return selections[:size]

    def _infer_archetype(self, leader: CardRecord) -> Optional[str]:
        if leader.family:
            return leader.family.split("/")[0].strip()
        if leader.color:
            return f"{leader.color} generic"
        return None

    def _build_prompt(self, leader: CardRecord, deck: DeckSchema) -> str:
        colors = _normalise_color_label(leader.color)
        focus = _extract_focus_from_ability(leader.ability) or self.rng.choice(FOCUS_DEFAULTS)
        template = self.rng.choice(PROMPT_TEMPLATES)
        base_prompt = template.format(
            leader=leader.name,
            colors=colors,
            focus=focus,
            opposition=self.rng.choice(OPPOSITION_TEMPLATES),
        )

        leader_ability = (leader.ability or "").strip() or "No special ability provided."
        leader_family = (leader.family or "").strip()
        leader_subtypes = [part.strip() for part in leader_family.replace("&", "/").split("/") if part.strip()]
        leader_colors = [
            part.strip() for part in (leader.color or "").replace("&", "/").split("/") if part.strip()
        ]

        context_lines = [
            f"Leader: {leader.name} ({leader.canonical_id})",
            f"Leader Ability: {leader_ability}",
            f"Leader Type: {', '.join(leader_subtypes) if leader_subtypes else 'None'}",
            f"Leader Colors: {', '.join(leader_colors) if leader_colors else 'Unknown'}",
            "",
            base_prompt,
        ]
        return "\n".join(context_lines)

    def _split_multivalue(self, value: Optional[str]) -> List[str]:
        if not value:
            return []
        parts = [part.strip() for part in value.replace("&", "/").split("/") if part.strip()]
        seen: Dict[str, None] = {}
        ordered: List[str] = []
        for part in parts:
            key = part.lower()
            if key not in seen:
                seen[key] = None
                ordered.append(part)
        return ordered

    def _ability_keywords(self, text: Optional[str]) -> Set[str]:
        if not text:
            return set()
        lowered = text.lower()
        keywords = {
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
        }
        detected = {keyword for keyword in keywords if keyword in lowered}
        return detected

    def _generate_card_rationale(
        self,
        card: CardRecord,
        leader: CardRecord,
        leader_subtypes: List[str],
        leader_colors: List[str],
        leader_keywords: Set[str],
    ) -> str:
        reasons: List[str] = []

        card_subtypes = self._split_multivalue(card.family)
        shared_subtypes = sorted({subtype for subtype in card_subtypes if subtype in leader_subtypes})
        if shared_subtypes:
            reasons.append(
                f"Shares {', '.join(shared_subtypes)} subtype(s) with {leader.name}, supporting archetype synergies.",
            )

        card_colors = self._split_multivalue(card.color)
        if leader_colors and card_colors and set(card_colors).issubset(set(leader_colors)):
            reasons.append("Matches the leader's color identity for consistent play.")

        card_keywords = self._ability_keywords(card.ability)
        keyword_overlap = sorted(card_keywords & leader_keywords)
        if keyword_overlap:
            reasons.append(
                f"Aligns with leader ability focus through {', '.join(keyword_overlap)} effects.",
            )
        elif card_keywords:
            reasons.append(
                f"Provides {', '.join(sorted(card_keywords))} utility that complements the leader's strategy.",
            )

        if not reasons:
            role = (card.type or "card").lower()
            reasons.append(f"Reliable {role} option that fits {leader.name}'s overall game plan.")

        return " ".join(reasons)

    def _build_synergy_map(self, card_ids: Sequence[str], max_links: int = 6) -> Dict[str, List[str]]:
        if not card_ids:
            return {}

        co_occurrence: Dict[str, Counter[str]] = defaultdict(Counter)
        cards = [card_id.upper() for card_id in card_ids]
        for index, card_id in enumerate(cards):
            for other in cards[index + 1 :]:
                if card_id == other:
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


def generate_synthetic_examples(
    repository: CardRepository,
    samples_per_leader: int = 3,
    deck_config: DeckConfig = DeckConfig(),
    prompt_config: PromptConfig = PromptConfig(),
    splits: TrainingSplits = TrainingSplits(),
    seed: int = 1234,
) -> List[PromptDeckExample]:
    generator = SyntheticDeckGenerator(
        repository=repository,
        deck_config=deck_config,
        prompt_config=prompt_config,
        random_seed=seed,
    )
    rng = random.Random(seed)
    examples: List[PromptDeckExample] = []

    split_thresholds = []
    cumulative = 0.0
    for split_name, proportion in splits.normalised().items():
        cumulative += proportion
        split_thresholds.append((cumulative, split_name))

    for leader in repository.leaders():
        for _ in range(samples_per_leader):
            example = generator.generate_example(leader)
            draw = rng.random()
            for threshold, split_name in split_thresholds:
                if draw <= threshold:
                    example.split = split_name
                    break
            examples.append(example)
    return examples


def write_examples_jsonl(examples: Iterable[PromptDeckExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for example in examples:
            fp.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")


def load_examples_jsonl(path: Path) -> List[PromptDeckExample]:
    examples: List[PromptDeckExample] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            examples.append(PromptDeckExample.from_record(record))
    return examples


def generate_and_export(
    data_root: Path,
    output_dir: Path = Path("ml/artifacts"),
    samples_per_leader: int = 3,
    seed: int = 1234,
    include_tournament_decks: bool = True,
    tournaments_dir: Optional[Path] = None,
    language: str = "en",
) -> Path:
    repository = CardRepository(data_root=data_root, language=language)
    examples = generate_synthetic_examples(
        repository=repository,
        samples_per_leader=samples_per_leader,
        seed=seed,
    )
    if include_tournament_decks:
        tournament_examples = load_tournament_examples(
            data_root=data_root,
            tournaments_dir=tournaments_dir,
            language=language,
            repository=repository,
        )
        examples.extend(tournament_examples)
    output_path = output_dir / "synthetic_prompt_deck.jsonl"
    write_examples_jsonl(examples, output_path)
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "data"
    path = generate_and_export(root)
    print(f"Synthetic dataset written to {path}")

