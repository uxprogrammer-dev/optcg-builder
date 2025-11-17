# Retraining Recommendations for Duplicate Card Issue

## Problem
The model generates 48-62 unique cards (all 1x) despite:
- Tournament decks averaging **16.3 unique cards** with only **2.1 cards at 1x** (12.8% of unique cards)
- Tournament decks have **9.0 cards at 4x** on average (55.3% of unique cards)
- Inference-time biasing with boosts of 300+ logits
- Anti-singleton loss during training

## Root Cause
The model learns local sequence patterns (via teacher forcing) rather than global frequency distributions. Even though training data has duplicates, deck shuffling randomizes order, so the model rarely sees the same card appear multiple times consecutively. The model has learned such a strong preference for diversity that even massive inference-time biases (300+ logits) cannot overcome it.

## Solution: Retrain with These Changes

### 1. Reduce Deck Shuffling During Training
**File**: `ml/datasets/tfdata.py` (line ~398)

**Current**: `deck_shuffle_rate = 0.2` (20% of decks are shuffled)

**Recommendation**: Reduce to `0.05` (5%) or `0.0` (no shuffling) so the model sees duplicates more often.

```python
# In make_tf_dataset function
deck_shuffle_rate: float = 0.05,  # Reduced from 0.2 - allow duplicates to appear consecutively
```

### 2. Increase Frequency Histogram Loss Weight
**File**: `ml/training/train.py`

**Current**: `freq_hist_weight = 50.0` (from your last training)

**Recommendation**: Increase to `100.0` or `150.0` to strongly penalize singleton-heavy decks.

```python
freq_hist_weight: float = 100.0,  # Increased from 50.0 - need stronger penalty
```

### 3. Increase Anti-Singleton Loss Penalties
**File**: `ml/training/train.py`

**Current**:
- `entropy_penalty = 1.0`
- `low_prob_penalty = 5.0`
- `low_prob_threshold = 0.25`

**Recommendation**: Increase penalties:
```python
entropy_penalty: float = 2.0,  # Increased from 1.0
low_prob_penalty: float = 10.0,  # Increased from 5.0
low_prob_threshold: float = 0.3,  # Increased from 0.25 (penalize more cards)
```

### 4. Verify Tournament Decks Are Included
**File**: `scripts/merge_tournament_decks.py`

Make sure tournament decks are being merged into the training dataset. The tournament decks have realistic duplicates (4x staples, 2x tech cards) that the model needs to learn from.

### 5. Training Command
```bash
python -m ml.training.train \
  --dataset data/training/decks_merged.jsonl \
  --output-dir models/run_$(date +%Y%m%d-%H%M%S) \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --freq-hist-weight 100.0 \
  --entropy-penalty 2.0 \
  --low-prob-penalty 10.0 \
  --low-prob-threshold 0.3 \
  --coverage-weight 0.05 \
  --disable-checkpoints  # If disk space is limited
```

## Expected Results
After retraining with these changes:
- Model should generate decks with realistic duplicates (4x staples, 2x tech cards)
- **Target: ~16 unique cards** (matching tournament deck average) instead of 48-62
- **Target: ~2 cards at 1x** (matching tournament deck average) instead of 48-62
- **Target: ~9 cards at 4x** (matching tournament deck average)
- Better alignment with tournament deck statistics

## Why This Should Work
1. **Reduced shuffling**: Model sees duplicates consecutively more often, learning that "same card → same card" is valid
2. **Stronger loss penalties**: Model is heavily penalized for generating singleton-heavy decks
3. **Tournament data**: Model learns from real decks with realistic card counts

## Alternative: Post-Processing (Not Recommended)
If retraining doesn't work, we could add post-processing to:
- Count card frequencies in generated deck
- Replace singleton cards with duplicates of high-frequency cards
- But this is a band-aid solution - the model should learn this naturally.

