# Next Steps After Training Models

You've successfully trained both the **deck generation model** and the **intent model**. Here's what to do next:

## 1. Evaluate Model Performance

### Evaluate Deck Generation Model

Run evaluation on the test set to verify model quality:

```bash
python -m ml.evaluation.evaluate \
  --dataset ml/artifacts/synthetic_prompt_deck.jsonl \
  --model models/run_20251114-151859/run_20251114-071943/deck_transformer.keras \
  --prompt-vocab models/run_20251114-151859/run_20251114-071943/vocab/prompt_vocabulary.txt \
  --card-vocab models/run_20251114-151859/run_20251114-071943/vocab/card_vocabulary.json \
  --split test \
  --decode-strategy beam \
  --beam-width 5 \
  --output ml/artifacts/eval_report.json
```

This will generate metrics including:
- Exact match rate
- Leader match rate
- Average deck overlap (Jaccard similarity)
- Legality rate
- Duplicate violation rate

### Test Intent Model Manually

Test the intent classifier with sample prompts:

```bash
python -m ml.inference.intent_classify \
  --prompt "I want to build a red rush deck with Roronoa Zoro" \
  --model models/intent_run_20251114-222759/intent_classifier.keras \
  --prompt-vocab models/intent_run_20251114-222759/vocab/prompt_vocabulary.txt \
  --color-vocab models/intent_run_20251114-222759/vocab/color_vocabulary.json \
  --keyword-vocab models/intent_run_20251114-222759/vocab/keyword_vocabulary.json \
  --leader-vocab models/intent_run_20251114-222759/vocab/leader_vocabulary.json \
  --strategy-vocab models/intent_run_20251114-222759/vocab/strategy_vocabulary.json \
  --leader-ids models/intent_run_20251114-222759/vocab/leader_ids.json
```

## 2. Update Environment Configuration

Create or update your `.env` file with paths to the new models:

```env
# Enable ML models
ML_MODEL_ENABLED=true
ML_INTENT_ENABLED=true

# Deck Generation Model
ML_MODEL_PATH=models/run_20251114-151859/run_20251114-071943/deck_transformer.keras
ML_DECODE_STRATEGY=beam
ML_BEAM_WIDTH=5
ML_LENGTH_PENALTY=0.7
ML_TIMEOUT_MS=15000

# Intent Model
ML_INTENT_MODEL_PATH=models/intent_run_20251114-222759/intent_classifier.keras
ML_INTENT_THRESHOLD=0.5
ML_INTENT_TOP_K_LEADERS=10
ML_INTENT_TOP_K_KEYWORDS=20

# Python path (adjust if needed)
ML_PYTHON_PATH=python
ML_DATA_ROOT=data
```

**Note:** The application automatically derives vocabulary paths from the model path, so you only need to set the model paths.

## 3. Test Models in Application

### Start the Application

```bash
npm run dev
```

### Test Deck Generation

1. Navigate to the web interface (usually `http://localhost:3001`)
2. Enter a prompt like: "Create a competitive red rush deck"
3. Select a leader (or let the model choose)
4. Generate a deck and verify it works correctly

### Test Intent Classification

The intent model is used automatically when suggesting leaders. Test it by:
1. Entering various prompts with different intents
2. Checking that leader suggestions match the prompt intent
3. Verifying colors, keywords, and strategies are correctly identified

## 4. Monitor and Iterate

- Check application logs for any errors or warnings
- Monitor model inference times
- Collect user feedback on generated decks
- Consider retraining if quality metrics are below expectations

## Quick Reference: Latest Models

- **Deck Model**: `models/run_20251114-151859/run_20251114-071943/deck_transformer.keras`
- **Intent Model**: `models/intent_run_20251114-222759/intent_classifier.keras`

## Troubleshooting

If models don't load:
1. Verify Python path is correct (`ML_PYTHON_PATH`)
2. Check that all vocabulary files exist in the model directories
3. Ensure TensorFlow/Keras dependencies are installed
4. Check application logs for specific error messages

If inference is slow:
- Try using `greedy` decode strategy instead of `beam`
- Reduce `ML_BEAM_WIDTH` for faster beam search
- Increase `ML_TIMEOUT_MS` if timeouts occur

