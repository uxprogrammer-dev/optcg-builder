"""
Script to analyze tournament deck sizes and count how many have exactly 50 cards (51 total with leader).
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict

# Add ml directory to path to import CardRepository
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.loader import CardRepository

def _coerce_quantity(value) -> int:
    """Coerce quantity to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0

def _resolve_card_id(card_id: str, repository: CardRepository):
    """Resolve card ID using repository."""
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

def analyze_tournament_decks(data_root: Path = Path("data")) -> None:
    """Analyze tournament deck sizes."""
    tournaments_dir = data_root / "tournaments"
    
    if not tournaments_dir.exists():
        print(f"Tournament directory not found: {tournaments_dir}")
        return
    
    # Load card repository to identify leaders
    repository = CardRepository(data_root=data_root, language="en")
    repository.load()
    
    deck_sizes = Counter()  # main_deck size (excluding leader)
    total_decks = 0
    decks_with_50_cards = 0
    decks_without_leader = 0
    decks_by_file: Dict[str, Dict] = {}
    
    for json_path in sorted(tournaments_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as e:
            print(f"Error reading {json_path.name}: {e}")
            continue
        
        if not isinstance(payload, list):
            print(f"Skipping {json_path.name}: not a list")
            continue
        
        file_counter = Counter()
        file_decks_with_50 = 0
        file_total = 0
        file_without_leader = 0
        
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            
            decklist = entry.get("decklist")
            if not isinstance(decklist, dict):
                continue
            
            main_deck_count = 0
            has_leader = False
            
            for raw_card_id, raw_quantity in decklist.items():
                if raw_card_id is None:
                    continue
                
                resolved_quantity = _coerce_quantity(raw_quantity)
                if resolved_quantity <= 0:
                    continue
                
                resolved_id, card_record = _resolve_card_id(str(raw_card_id), repository)
                
                if card_record is None:
                    # Unknown card - count it as main deck card
                    main_deck_count += resolved_quantity
                    continue
                
                if card_record.is_leader:
                    has_leader = True
                    # Leader doesn't count toward main_deck
                    continue
                
                # Regular card - add to main_deck count
                main_deck_count += resolved_quantity
            
            if not has_leader:
                file_without_leader += 1
                decks_without_leader += 1
            
            deck_sizes[main_deck_count] += 1
            file_counter[main_deck_count] += 1
            total_decks += 1
            file_total += 1
            
            # Check if main_deck has exactly 50 cards
            if main_deck_count == 50:
                decks_with_50_cards += 1
                file_decks_with_50 += 1
        
        decks_by_file[json_path.name] = {
            'counter': file_counter,
            'total': file_total,
            'with_50_cards': file_decks_with_50,
            'without_leader': file_without_leader,
        }
    
    # Print summary
    print("=" * 80)
    print("TOURNAMENT DECK SIZE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal decks analyzed: {total_decks}")
    print(f"Decks with exactly 50 cards in main deck: {decks_with_50_cards}")
    print(f"Decks without leader identified: {decks_without_leader}")
    print(f"Percentage with exactly 50 cards: {decks_with_50_cards / total_decks * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("MAIN DECK SIZE DISTRIBUTION (excluding leader)")
    print("=" * 80)
    for size in sorted(deck_sizes.keys()):
        count = deck_sizes[size]
        percentage = count / total_decks * 100
        marker = " <-- EXACTLY 50 CARDS" if size == 50 else ""
        print(f"  {size:3d} cards: {count:5d} decks ({percentage:5.2f}%){marker}")
    
    print("\n" + "=" * 80)
    print("BREAKDOWN BY FILE")
    print("=" * 80)
    for filename, stats in sorted(decks_by_file.items()):
        print(f"\n{filename}:")
        print(f"  Total decks: {stats['total']}")
        print(f"  Decks with exactly 50 cards: {stats['with_50_cards']} ({stats['with_50_cards'] / stats['total'] * 100:.2f}%)")
        print(f"  Decks without leader: {stats['without_leader']}")
        if stats['counter']:
            print(f"  Size distribution:")
            for size in sorted(stats['counter'].keys()):
                count = stats['counter'][size]
                marker = " <-- 50" if size == 50 else ""
                print(f"    {size:3d} cards: {count:4d}{marker}")

if __name__ == "__main__":
    analyze_tournament_decks()