# Prompt→Deck Generation Workflow

This directory contains the end-to-end TensorFlow/Keras pipeline for training a model that maps natural-language prompts to legal One Piece card decks.

## 1. Environment Setup

```bash
python -m venv .venv && source .venv/Scripts/activate  # Windows Git Bash example
pip install -r ml/requirements.txt
```

> TensorFlow is optional for dataset generation. Training and evaluation require the dependencies above.

## 2. Card Data Profiling

Summaries of the JSON exports live in `ml/artifacts/card_profile_en.json`.

```bash
python -m ml.data.profiling
```

The schema conventions (card record, deck representation, prompt examples) are documented in `ml/docs/schema.md`.

## 3. Synthetic Dataset Generation

Produce prompt→deck pairs and optional vocabularies:

```bash
# Combined dataset + per-split JSONL files
python -m ml.pipelines.prepare_data

# Include --export-tf-assets if TensorFlow is installed to emit vocabularies
python -m ml.pipelines.prepare_data --export-tf-assets
```

Artifacts:

- `ml/artifacts/synthetic_prompt_deck.jsonl` – master dataset.
- `ml/artifacts/synthetic_prompt_deck_{train,val,test}.jsonl` – split subsets.
- `ml/artifacts/vocab/` – prompt vocabulary + card id vocabulary (when `--export-tf-assets` is set).

### tf.data Input Pipeline

`ml/datasets/tfdata.py` exposes:

- `build_card_vocabulary(...)`
- `create_text_vectorizer(...)` / `create_text_vectorizer_from_vocab(...)`
- `make_tf_dataset(...)` – batches, caches, shuffles, and performs on-the-fly prompt paraphrasing + deck-order augmentation.

## 4. Model Architecture

`ml/models/transformer.py` implements an encoder-decoder Transformer with:

- Positional encodings and tied embedding dimensions.
- Multi-head self- and cross-attention layers.
- Coverage loss regularisation (`coverage_loss_weight`) to discourage under-attending to encoder tokens.

Instantiate via:

```python
from ml.models import TransformerConfig, build_deck_transformer
model = build_deck_transformer(
    prompt_vocab_size=len(prompt_vocab),
    deck_vocab_size=len(card_to_index),
    prompt_sequence_length=PromptConfig().max_length,
    deck_sequence_length=DeckConfig().max_total_cards + 2,
    pad_token_id=card_to_index[DeckConfig().pad_token],
    config=TransformerConfig(),
)
```

## 5. Training

Run supervised training with masked cross-entropy and accuracy:

```bash
python -m ml.training.train \
  --dataset ml/artifacts/synthetic_prompt_deck.jsonl \
  --output-dir models \
  --epochs 20 \
  --batch-size 48 \
  --learning-rate 3e-4
```

Outputs (`models/run_*/`):

- `deck_transformer.keras` – saved weights.
- `history.json` & `test_metrics.json`.
- `checkpoints/` – per-epoch checkpoints.
- `logs/` – TensorBoard summaries.
- `vocab/` – prompt + card vocabularies used for the run.

Losses (`ml/training/losses.py`) provide padding-aware cross-entropy and accuracy helpers.

## 6. Inference & Evaluation

### Generation Helpers

`ml/inference/generate.py` offers:

- `greedy_generate(...)`
- `beam_search_generate(...)` (length-penalised beam search).

For ad-hoc inspection outside of NestJS, call the CLI wrapper:

```bash
python -m ml.inference.generate_deck \
  --prompt "Red rush deck built around Roronoa Zoro" \
  --model models/run_YYYYMMDD-HHMMSS/deck_transformer.keras \
  --prompt-vocab models/run_YYYYMMDD-HHMMSS/vocab/prompt_vocabulary.txt \
  --card-vocab models/run_YYYYMMDD-HHMMSS/vocab/card_vocabulary.json
```

### Evaluation Script

Compare predictions with reference decks:

```bash
python -m ml.evaluation.evaluate \
  --dataset ml/artifacts/synthetic_prompt_deck.jsonl \
  --model models/run_YYYYMMDD-HHMMSS/deck_transformer.keras \
  --prompt-vocab models/run_YYYYMMDD-HHMMSS/vocab/prompt_vocabulary.txt \
  --card-vocab models/run_YYYYMMDD-HHMMSS/vocab/card_vocabulary.json \
  --split test \
  --decode-strategy beam \
  --beam-width 5
```

Reports land in `ml/artifacts/eval_report.json` by default and include:

- Exact-match rate
- Leader-match rate
- Deck overlap (Jaccard)
- Legality/duplicate diagnostics
- Unknown-card counts

Supporting functions in `ml/evaluation/metrics.py` convert sequences back to `DeckSchema` objects and enforce duplicate limits.

## 7. Directory Reference

- `ml/data/` – schema definitions, repository loader, profiling helpers.
- `ml/datasets/` – synthetic data generation + TensorFlow dataset builders.
- `ml/models/` – Keras Transformer architecture.
- `ml/training/` – training loop, losses, and run outputs.
- `ml/inference/` – prompt decoding utilities.
- `ml/evaluation/` – metrics & CLI evaluation.
- `ml/docs/` – design notes (`schema.md`).

## 8. Next Steps

- Replace synthetic data with curated prompt/deck pairs as they become available.
- Extend legality checks with full OPTCG rules (color pair legality, leader-specific constraints).
- Integrate API-facing serving layer once model quality is validated.

