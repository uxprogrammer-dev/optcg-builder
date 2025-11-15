"""
Test script to verify deck normalization works correctly.
"""

import sys
from pathlib import Path

# Add ml directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.datasets.tournament import load_tournament_examples
from collections import Counter

def main():
    data_root = Path("data")
    examples = load_tournament_examples(data_root)
    
    sizes = [len(ex.deck.main_deck) for ex in examples]
    c = Counter(sizes)
    
    print("=" * 80)
    print("DECK NORMALIZATION TEST")
    print("=" * 80)
    print(f"\nTotal decks loaded: {len(examples)}")
    print(f"Decks with exactly 50 cards: {c[50]}")
    print(f"Percentage with exactly 50 cards: {c[50]/len(examples)*100:.2f}%")
    
    print("\nDeck size distribution:")
    for size in sorted(c.keys()):
        count = c[size]
        percentage = count / len(examples) * 100
        marker = " ✓" if size == 50 else ""
        print(f"  {size:3d} cards: {count:5d} decks ({percentage:5.2f}%){marker}")
    
    # Check if all decks are exactly 50
    if c[50] == len(examples):
        print("\n✓ SUCCESS: All decks have exactly 50 cards!")
    else:
        non_50 = len(examples) - c[50]
        print(f"\n⚠ WARNING: {non_50} decks do not have exactly 50 cards")
        print("   These decks may need additional normalization.")

if __name__ == "__main__":
    main()

