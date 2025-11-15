#!/usr/bin/env python3
"""
Script to populate archetypes.json, card-tiers.json, and leader-strategies.json
by analyzing tournament deck data and card information.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Configuration
DATA_ROOT = Path("data")
CARDS_DIR = DATA_ROOT / "cards" / "en"
TOURNAMENTS_DIR = DATA_ROOT / "tournaments"
LEADERS_FILE = DATA_ROOT / "leaders.json"
OUTPUT_DIR = DATA_ROOT

# Placement scoring (higher = better)
PLACEMENT_SCORES = {
    "1st Place": 10,
    "2nd Place": 8,
    "3rd Place": 7,
    "T4": 6,
    "T8": 5,
    "T16": 4,
    "T32": 3,
    "Top": 5,
    "Winner": 10,
}


def load_json_file(path: Path) -> Optional[dict | list]:
    """Load a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def load_all_cards() -> Dict[str, dict]:
    """Load all cards from all set files."""
    cards = {}
    for json_file in sorted(CARDS_DIR.glob("*.json")):
        data = load_json_file(json_file)
        if isinstance(data, list):
            for card in data:
                if "id" in card:
                    cards[card["id"].upper()] = card
    return cards


def load_tournament_decks() -> List[dict]:
    """Load all tournament deck files."""
    decks = []
    for json_file in sorted(TOURNAMENTS_DIR.glob("*.json")):
        data = load_json_file(json_file)
        if isinstance(data, list):
            decks.extend(data)
    return decks


def get_placement_score(placement: str) -> int:
    """Get numeric score for placement."""
    placement_lower = placement.lower()
    for key, score in PLACEMENT_SCORES.items():
        if key.lower() in placement_lower:
            return score
    return 1  # Default score for unknown placements


def analyze_card_tiers(decks: List[dict], cards: Dict[str, dict]) -> Dict:
    """Analyze tournament decks to determine card tiers."""
    # Track card usage: card_id -> (total_count, weighted_score, leader_usage)
    card_stats: Dict[str, Tuple[int, float, Dict[str, float]]] = defaultdict(
        lambda: (0, 0.0, defaultdict(float))
    )

    for deck in decks:
        decklist = deck.get("decklist", {})
        placement = deck.get("placement", "")
        leader_id = None

        # Find leader
        for card_id, quantity in decklist.items():
            card = cards.get(card_id.upper())
            if card and card.get("type", "").upper() == "LEADER":
                leader_id = card_id.upper()
                break

        if not leader_id:
            continue

        placement_score = get_placement_score(placement)

        # Count card usage
        for card_id, quantity in decklist.items():
            card_id_upper = card_id.upper()
            if card_id_upper == leader_id:
                continue

            card = cards.get(card_id_upper)
            if not card or card.get("type", "").upper() == "LEADER":
                continue

            qty = int(quantity) if isinstance(quantity, (int, str)) else 0
            if qty > 0:
                total_count, weighted_score, leader_usage = card_stats[card_id_upper]
                total_count += qty
                weighted_score += qty * placement_score
                leader_usage[leader_id] += qty * placement_score
                card_stats[card_id_upper] = (total_count, weighted_score, leader_usage)

    # Calculate tiers
    # S tier: Top 5% by weighted score, appears in 3+ top placements
    # A tier: Top 15% by weighted score, appears in 2+ top placements
    # B tier: Top 30% by weighted score, appears in 1+ top placement
    # C tier: Everything else

    sorted_cards = sorted(
        card_stats.items(), key=lambda x: x[1][1], reverse=True
    )  # Sort by weighted score

    total_cards = len(sorted_cards)
    s_tier_count = max(5, int(total_cards * 0.05))
    a_tier_count = max(15, int(total_cards * 0.15))
    b_tier_count = max(30, int(total_cards * 0.30))

    tiers = {"S": [], "A": [], "B": [], "C": []}

    for idx, (card_id, (total_count, weighted_score, leader_usage)) in enumerate(
        sorted_cards
    ):
        if idx < s_tier_count:
            tiers["S"].append(card_id)
        elif idx < a_tier_count:
            tiers["A"].append(card_id)
        elif idx < b_tier_count:
            tiers["B"].append(card_id)
        else:
            tiers["C"].append(card_id)

    # Build leader-specific tiers
    leader_specific: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {"S": [], "A": [], "B": [], "C": []}
    )

    for leader_id in set(
        leader
        for stats in card_stats.values()
        for leader in stats[2].keys()
    ):
        leader_cards = [
            (card_id, leader_usage.get(leader_id, 0))
            for card_id, (_, _, leader_usage) in card_stats.items()
            if leader_id in leader_usage
        ]
        leader_cards.sort(key=lambda x: x[1], reverse=True)

        leader_total = len(leader_cards)
        leader_s_tier = max(3, int(leader_total * 0.1))
        leader_a_tier = max(8, int(leader_total * 0.25))
        leader_b_tier = max(15, int(leader_total * 0.50))

        for idx, (card_id, _) in enumerate(leader_cards):
            if idx < leader_s_tier:
                leader_specific[leader_id]["S"].append(card_id)
            elif idx < leader_a_tier:
                leader_specific[leader_id]["A"].append(card_id)
            elif idx < leader_b_tier:
                leader_specific[leader_id]["B"].append(card_id)
            else:
                leader_specific[leader_id]["C"].append(card_id)

    return {
        "version": "1.0",
        "tiers": {
            "S": {"description": "Top tier - Must include in competitive decks", "cards": tiers["S"]},
            "A": {"description": "High tier - Strong competitive cards", "cards": tiers["A"]},
            "B": {"description": "Good tier - Solid role players", "cards": tiers["B"]},
            "C": {"description": "Average tier - Situational or niche cards", "cards": tiers["C"]},
        },
        "leaderSpecific": dict(leader_specific),
    }


def analyze_leader_strategies(
    decks: List[dict], cards: Dict[str, dict]
) -> List[dict]:
    """Analyze tournament decks to extract leader strategies."""
    leader_decks: Dict[str, List[dict]] = defaultdict(list)

    # Group decks by leader
    for deck in decks:
        decklist = deck.get("decklist", {})
        leader_id = None

        for card_id, quantity in decklist.items():
            card = cards.get(card_id.upper())
            if card and card.get("type", "").upper() == "LEADER":
                leader_id = card_id.upper()
                break

        if leader_id:
            leader_decks[leader_id].append(deck)

    strategies = []

    for leader_id, leader_deck_list in leader_decks.items():
        if len(leader_deck_list) < 2:  # Need at least 2 decks to analyze
            continue

        leader_card = cards.get(leader_id)
        if not leader_card:
            continue

        # Analyze card usage patterns
        card_frequency: Dict[str, int] = Counter()
        card_quantities: Dict[str, List[int]] = defaultdict(list)
        deck_names: List[str] = []

        for deck in leader_deck_list:
            decklist = deck.get("decklist", {})
            deck_names.append(deck.get("deck_name", ""))
            for card_id, quantity in decklist.items():
                card_id_upper = card_id.upper()
                if card_id_upper == leader_id:
                    continue
                card = cards.get(card_id_upper)
                if card and card.get("type", "").upper() != "LEADER":
                    qty = int(quantity) if isinstance(quantity, (int, str)) else 0
                    if qty > 0:
                        card_frequency[card_id_upper] += 1
                        card_quantities[card_id_upper].append(qty)

        # Get most common cards (appear in 50%+ of decks)
        min_decks = max(2, len(leader_deck_list) * 0.5)
        key_cards = [
            (card_id, sum(card_quantities[card_id]) / len(card_quantities[card_id]))
            for card_id, count in card_frequency.items()
            if count >= min_decks
        ]
        key_cards.sort(key=lambda x: x[1], reverse=True)

        # Determine archetype from deck names
        archetype = "Midrange"  # Default
        deck_name_words = Counter()
        for name in deck_names:
            if name:
                words = re.findall(r"\w+", name.lower())
                deck_name_words.update(words)

        common_words = [word for word, count in deck_name_words.most_common(3) if count >= 2]
        if "aggro" in common_words or "rush" in common_words:
            archetype = "Aggro"
        elif "control" in common_words:
            archetype = "Control"
        elif "combo" in common_words:
            archetype = "Combo"

        # Analyze cost curve
        all_costs = []
        for deck in leader_deck_list:
            decklist = deck.get("decklist", {})
            for card_id, quantity in decklist.items():
                card_id_upper = card_id.upper()
                if card_id_upper == leader_id:
                    continue
                card = cards.get(card_id_upper)
                if card:
                    cost = card.get("cost")
                    if isinstance(cost, (int, str)):
                        try:
                            cost_int = int(cost)
                            qty = int(quantity) if isinstance(quantity, (int, str)) else 0
                            all_costs.extend([cost_int] * qty)
                        except:
                            pass

        cost_distribution = Counter(all_costs)
        total_cards = sum(cost_distribution.values())
        if total_cards > 0:
            cost_0_2 = sum(count for cost, count in cost_distribution.items() if 0 <= cost <= 2) / total_cards
            cost_3_4 = sum(count for cost, count in cost_distribution.items() if 3 <= cost <= 4) / total_cards
            cost_5_6 = sum(count for cost, count in cost_distribution.items() if 5 <= cost <= 6) / total_cards
            cost_7_plus = sum(count for cost, count in cost_distribution.items() if cost >= 7) / total_cards
        else:
            cost_0_2 = cost_3_4 = cost_5_6 = cost_7_plus = 0.25

        # Analyze card types
        type_counts = Counter()
        for deck in leader_deck_list:
            decklist = deck.get("decklist", {})
            for card_id, quantity in decklist.items():
                card_id_upper = card_id.upper()
                if card_id_upper == leader_id:
                    continue
                card = cards.get(card_id_upper)
                if card:
                    card_type = card.get("type", "").upper()
                    qty = int(quantity) if isinstance(quantity, (int, str)) else 0
                    type_counts[card_type] += qty

        total_type_cards = sum(type_counts.values())
        if total_type_cards > 0:
            characters_ratio = type_counts.get("CHARACTER", 0) / total_type_cards
            events_ratio = type_counts.get("EVENT", 0) / total_type_cards
            stages_ratio = type_counts.get("STAGE", 0) / total_type_cards
        else:
            characters_ratio = 0.65
            events_ratio = 0.25
            stages_ratio = 0.10

        # Generate strategy description
        leader_ability = leader_card.get("ability", "")
        leader_colors = leader_card.get("color", "").split("/")
        leader_family = leader_card.get("family", "").split("/")

        strategy_parts = []
        if "draw" in leader_ability.lower() or "add to hand" in leader_ability.lower():
            strategy_parts.append("Generate card advantage")
        if "k.o." in leader_ability.lower() or "destroy" in leader_ability.lower():
            strategy_parts.append("Control the board with removal")
        if "don" in leader_ability.lower() or "rest" in leader_ability.lower():
            strategy_parts.append("Manage resources efficiently")
        if "power" in leader_ability.lower() and "+" in leader_ability:
            strategy_parts.append("Build board presence")

        if not strategy_parts:
            strategy_parts.append("Build a synergistic deck around the leader ability")

        strategy = f"{archetype} deck that focuses on {', '.join(strategy_parts[:2])}."

        # Build key cards list
        key_cards_list = []
        for card_id, avg_qty in key_cards[:5]:  # Top 5 key cards
            card = cards.get(card_id)
            if card:
                card_name = card.get("name", card_id)
                card_ability = card.get("ability", "")
                
                # Determine role
                role = "engine"
                if "k.o." in card_ability.lower() or "destroy" in card_ability.lower():
                    role = "removal"
                elif "draw" in card_ability.lower() or "search" in card_ability.lower():
                    role = "engine"
                elif int(card.get("cost", 0)) >= 6:
                    role = "finisher"
                elif "blocker" in card_ability.lower():
                    role = "defense"

                rationale = f"Core {role} card"
                if card_ability:
                    rationale += f" that {card_ability[:100]}"
                else:
                    rationale += f" for {archetype.lower()} strategy"

                key_cards_list.append({
                    "cardId": card_id,
                    "role": role,
                    "rationale": rationale,
                })

        # Win conditions
        win_conditions = []
        if archetype == "Aggro":
            win_conditions = ["Apply early pressure", "Use rush effects to push damage", "Maintain board presence"]
        elif archetype == "Control":
            win_conditions = ["Control the board with removal", "Generate card advantage", "Finish with high-cost threats"]
        else:
            win_conditions = ["Build board presence", "Use leader ability effectively", "Outvalue opponent"]

        strategies.append({
            "leaderId": leader_id,
            "leaderName": leader_card.get("name", leader_id),
            "archetype": archetype,
            "strategy": strategy,
            "keyCards": key_cards_list,
            "winConditions": win_conditions,
            "typicalRatios": {
                "characters": round(characters_ratio, 2),
                "events": round(events_ratio, 2),
                "stages": round(stages_ratio, 2),
            },
            "costCurve": {
                "0-2": round(cost_0_2, 2),
                "3-4": round(cost_3_4, 2),
                "5-6": round(cost_5_6, 2),
                "7+": round(cost_7_plus, 2),
            },
        })

    return strategies


def analyze_archetypes(decks: List[dict], cards: Dict[str, dict]) -> List[dict]:
    """Analyze tournament decks to identify archetypes."""
    archetype_decks: Dict[str, List[dict]] = defaultdict(list)

    # Group by deck name (normalized)
    for deck in decks:
        deck_name = deck.get("deck_name", "").strip()
        if deck_name:
            # Normalize name
            normalized = re.sub(r"[^a-zA-Z0-9\s]", "", deck_name.lower())
            normalized = re.sub(r"\s+", " ", normalized).strip()
            archetype_decks[normalized].append(deck)

    archetypes = []

    for archetype_name, archetype_deck_list in archetype_decks.items():
        if len(archetype_deck_list) < 2:  # Need at least 2 decks
            continue

        # Analyze common characteristics
        all_subtypes: Set[str] = set()
        all_colors: Set[str] = set()
        common_cards: Counter = Counter()
        deck_names = []

        for deck in archetype_deck_list:
            decklist = deck.get("decklist", {})
            deck_names.append(deck.get("deck_name", ""))
            leader_id = None

            for card_id, quantity in decklist.items():
                card = cards.get(card_id.upper())
                if card:
                    if card.get("type", "").upper() == "LEADER":
                        leader_id = card_id.upper()
                    else:
                        family = card.get("family", "")
                        if family:
                            all_subtypes.update(family.split("/"))
                        color = card.get("color", "")
                        if color:
                            all_colors.update(color.split("/"))
                        qty = int(quantity) if isinstance(quantity, (int, str)) else 0
                        if qty >= 3:  # Only count cards with 3+ copies
                            common_cards[card_id.upper()] += 1

        # Get most common cards (appear in 50%+ of decks)
        min_decks = max(2, len(archetype_deck_list) * 0.5)
        key_cards = [
            card_id for card_id, count in common_cards.items() if count >= min_decks
        ][:5]

        # Determine strategy from deck names and cards
        strategy = f"Archetype focused on {', '.join(list(all_subtypes)[:2]) if all_subtypes else 'synergy'}"

        # Determine typical ratios
        type_counts = Counter()
        for deck in archetype_deck_list:
            decklist = deck.get("decklist", {})
            for card_id, quantity in decklist.items():
                card = cards.get(card_id.upper())
                if card and card.get("type", "").upper() != "LEADER":
                    card_type = card.get("type", "").upper()
                    qty = int(quantity) if isinstance(quantity, (int, str)) else 0
                    type_counts[card_type] += qty

        total = sum(type_counts.values())
        if total > 0:
            characters_ratio = type_counts.get("CHARACTER", 0) / total
            events_ratio = type_counts.get("EVENT", 0) / total
            stages_ratio = type_counts.get("STAGE", 0) / total
        else:
            characters_ratio = 0.65
            events_ratio = 0.25
            stages_ratio = 0.10

        archetypes.append({
            "id": archetype_name.replace(" ", "-").lower(),
            "name": archetype_deck_list[0].get("deck_name", archetype_name),
            "description": strategy,
            "keySubtypes": list(all_subtypes)[:5],
            "keyColors": list(all_colors)[:3],
            "strategy": strategy,
            "keyCards": key_cards[:5],
            "typicalRatios": {
                "characters": round(characters_ratio, 2),
                "events": round(events_ratio, 2),
                "stages": round(stages_ratio, 2),
            },
        })

    return archetypes


def main():
    """Main function to populate all JSON files."""
    print("Loading card data...")
    cards = load_all_cards()
    print(f"Loaded {len(cards)} cards")

    print("Loading tournament decks...")
    decks = load_tournament_decks()
    print(f"Loaded {len(decks)} tournament decks")

    print("\nAnalyzing card tiers...")
    tiers_data = analyze_card_tiers(decks, cards)
    tiers_file = OUTPUT_DIR / "meta" / "card-tiers.json"
    tiers_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tiers_file, "w", encoding="utf-8") as f:
        json.dump(tiers_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote {len(tiers_data['tiers']['S'])} S-tier, {len(tiers_data['tiers']['A'])} A-tier cards to {tiers_file}")

    print("\nAnalyzing leader strategies...")
    strategies = analyze_leader_strategies(decks, cards)
    strategies_file = OUTPUT_DIR / "strategies" / "leader-strategies.json"
    strategies_file.parent.mkdir(parents=True, exist_ok=True)
    strategies_data = {"version": "1.0", "strategies": strategies}
    with open(strategies_file, "w", encoding="utf-8") as f:
        json.dump(strategies_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote {len(strategies)} leader strategies to {strategies_file}")

    print("\nAnalyzing archetypes...")
    archetypes = analyze_archetypes(decks, cards)
    archetypes_file = OUTPUT_DIR / "archetypes" / "archetypes.json"
    archetypes_file.parent.mkdir(parents=True, exist_ok=True)
    archetypes_data = {"version": "1.0", "archetypes": archetypes}
    with open(archetypes_file, "w", encoding="utf-8") as f:
        json.dump(archetypes_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote {len(archetypes)} archetypes to {archetypes_file}")

    print("\n✓ All data files populated successfully!")


if __name__ == "__main__":
    main()