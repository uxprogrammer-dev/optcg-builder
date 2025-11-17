#!/usr/bin/env python3
"""
Analyze duplicate patterns in tournament decks to understand expected card counts.
"""
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List

def analyze_tournament_decks(data_root: Path = Path("data")) -> None:
    """Analyze duplicate patterns in tournament decks."""
    tournaments_dir = data_root / "tournaments"
    if not tournaments_dir.exists():
        print(f"Tournament directory not found: {tournaments_dir}")
        return
    
    all_deck_stats = []
    total_decks = 0
    
    for json_file in sorted(tournaments_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            continue
        
        for entry in data:
            decklist = entry.get("decklist", {})
            if not isinstance(decklist, dict):
                continue
            
            # Count cards (excluding leader)
            card_counts = Counter()
            for card_id, quantity in decklist.items():
                if isinstance(quantity, int) and quantity > 0:
                    # Normalize card ID (remove variant suffixes)
                    base_id = card_id.upper().split("_")[0]
                    card_counts[base_id] += quantity
            
            if len(card_counts) == 0:
                continue
            
            total_decks += 1
            
            # Calculate statistics
            total_cards = sum(card_counts.values())
            unique_cards = len(card_counts)
            cards_1x = sum(1 for count in card_counts.values() if count == 1)
            cards_2x = sum(1 for count in card_counts.values() if count == 2)
            cards_3x = sum(1 for count in card_counts.values() if count == 3)
            cards_4x = sum(1 for count in card_counts.values() if count == 4)
            
            # Count distribution
            count_distribution = Counter(card_counts.values())
            
            all_deck_stats.append({
                "total_cards": total_cards,
                "unique_cards": unique_cards,
                "cards_1x": cards_1x,
                "cards_2x": cards_2x,
                "cards_3x": cards_3x,
                "cards_4x": cards_4x,
                "count_distribution": count_distribution,
            })
    
    if total_decks == 0:
        print("No tournament decks found!")
        return
    
    print(f"\n=== Tournament Deck Duplicate Analysis ===")
    print(f"Total decks analyzed: {total_decks}\n")
    
    # Aggregate statistics
    avg_total_cards = sum(s["total_cards"] for s in all_deck_stats) / len(all_deck_stats)
    avg_unique_cards = sum(s["unique_cards"] for s in all_deck_stats) / len(all_deck_stats)
    avg_1x = sum(s["cards_1x"] for s in all_deck_stats) / len(all_deck_stats)
    avg_2x = sum(s["cards_2x"] for s in all_deck_stats) / len(all_deck_stats)
    avg_3x = sum(s["cards_3x"] for s in all_deck_stats) / len(all_deck_stats)
    avg_4x = sum(s["cards_4x"] for s in all_deck_stats) / len(all_deck_stats)
    
    print(f"Average cards per deck: {avg_total_cards:.1f}")
    print(f"Average unique cards: {avg_unique_cards:.1f}")
    print(f"\nAverage card count distribution:")
    print(f"  1x cards: {avg_1x:.1f} ({avg_1x/avg_unique_cards*100:.1f}% of unique cards)")
    print(f"  2x cards: {avg_2x:.1f} ({avg_2x/avg_unique_cards*100:.1f}% of unique cards)")
    print(f"  3x cards: {avg_3x:.1f} ({avg_3x/avg_unique_cards*100:.1f}% of unique cards)")
    print(f"  4x cards: {avg_4x:.1f} ({avg_4x/avg_unique_cards*100:.1f}% of unique cards)")
    
    # Show distribution of 1x cards
    decks_by_1x = Counter(s["cards_1x"] for s in all_deck_stats)
    print(f"\nDistribution of 1x cards per deck:")
    for num_1x in sorted(decks_by_1x.keys()):
        count = decks_by_1x[num_1x]
        pct = count / total_decks * 100
        print(f"  {num_1x} 1x cards: {count} decks ({pct:.1f}%)")
    
    # Show examples of decks with many 1x cards vs few 1x cards
    sorted_by_1x = sorted(all_deck_stats, key=lambda s: s["cards_1x"])
    print(f"\nDecks with FEWEST 1x cards (top 5):")
    for i, stats in enumerate(sorted_by_1x[:5]):
        print(f"  {i+1}. {stats['cards_1x']} 1x, {stats['cards_2x']} 2x, {stats['cards_3x']} 3x, {stats['cards_4x']} 4x (total: {stats['total_cards']} cards, {stats['unique_cards']} unique)")
    
    print(f"\nDecks with MOST 1x cards (top 5):")
    for i, stats in enumerate(sorted_by_1x[-5:]):
        print(f"  {i+1}. {stats['cards_1x']} 1x, {stats['cards_2x']} 2x, {stats['cards_3x']} 3x, {stats['cards_4x']} 4x (total: {stats['total_cards']} cards, {stats['unique_cards']} unique)")

if __name__ == "__main__":
    analyze_tournament_decks()

