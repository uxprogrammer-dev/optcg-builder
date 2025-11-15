#!/bin/bash

# Script to evaluate the newly trained models
# Usage: ./scripts/evaluate_models.sh

set -e

echo "=== Evaluating Deck Generation Model ==="
echo ""

DECK_MODEL="models/run_20251114-151859/run_20251114-071943/deck_transformer.keras"
DECK_PROMPT_VOCAB="models/run_20251114-151859/run_20251114-071943/vocab/prompt_vocabulary.txt"
DECK_CARD_VOCAB="models/run_20251114-151859/run_20251114-071943/vocab/card_vocabulary.json"

if [ ! -f "$DECK_MODEL" ]; then
    echo "ERROR: Deck model not found at $DECK_MODEL"
    echo "Please update the path in this script to point to your latest deck model"
    exit 1
fi

python -m ml.evaluation.evaluate \
  --dataset ml/artifacts/synthetic_prompt_deck.jsonl \
  --model "$DECK_MODEL" \
  --prompt-vocab "$DECK_PROMPT_VOCAB" \
  --card-vocab "$DECK_CARD_VOCAB" \
  --split test \
  --decode-strategy beam \
  --beam-width 5 \
  --output ml/artifacts/eval_report.json

echo ""
echo "=== Evaluation complete! Report saved to ml/artifacts/eval_report.json ==="
echo ""
echo "=== Testing Intent Model with Sample Prompt ==="
echo ""

INTENT_MODEL="models/intent_run_20251114-222759/intent_classifier.keras"
INTENT_VOCAB_DIR="models/intent_run_20251114-222759/vocab"

if [ ! -f "$INTENT_MODEL" ]; then
    echo "ERROR: Intent model not found at $INTENT_MODEL"
    echo "Please update the path in this script to point to your latest intent model"
    exit 1
fi

python -m ml.inference.intent_classify \
  --prompt "I want to build a red rush deck with Roronoa Zoro" \
  --model "$INTENT_MODEL" \
  --prompt-vocab "$INTENT_VOCAB_DIR/prompt_vocabulary.txt" \
  --color-vocab "$INTENT_VOCAB_DIR/color_vocabulary.json" \
  --keyword-vocab "$INTENT_VOCAB_DIR/keyword_vocabulary.json" \
  --leader-vocab "$INTENT_VOCAB_DIR/leader_vocabulary.json" \
  --strategy-vocab "$INTENT_VOCAB_DIR/strategy_vocabulary.json" \
  --leader-ids "$INTENT_VOCAB_DIR/leader_ids.json"

echo ""
echo "=== Intent model test complete! ==="

