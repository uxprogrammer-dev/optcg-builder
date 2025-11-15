#!/bin/bash

# Quick test script for deck generation model
# Usage: ./scripts/test_deck_generation.sh "your prompt here"

set -e

PROMPT="${1:-Create a competitive red rush deck}"

echo "=== Testing Deck Generation Model ==="
echo "Prompt: $PROMPT"
echo ""

DECK_MODEL="models/run_20251114-151859/run_20251114-071943/deck_transformer.keras"
DECK_PROMPT_VOCAB="models/run_20251114-151859/run_20251114-071943/vocab/prompt_vocabulary.txt"
DECK_CARD_VOCAB="models/run_20251114-151859/run_20251114-071943/vocab/card_vocabulary.json"

if [ ! -f "$DECK_MODEL" ]; then
    echo "ERROR: Deck model not found at $DECK_MODEL"
    echo "Please update the path in this script to point to your latest deck model"
    exit 1
fi

python -m ml.inference.generate_deck \
  --prompt "$PROMPT" \
  --model "$DECK_MODEL" \
  --prompt-vocab "$DECK_PROMPT_VOCAB" \
  --card-vocab "$DECK_CARD_VOCAB" \
  --decode-strategy beam \
  --beam-width 5

echo ""
echo "=== Deck generation complete! ==="

