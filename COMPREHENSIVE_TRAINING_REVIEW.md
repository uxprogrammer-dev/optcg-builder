# Comprehensive Review: Model Training & Generation for Competitive Tournament Decks

## Executive Summary

**The model generates 48-62 unique cards (all 1x) instead of tournament-style decks (16.3 unique, 2.1 at 1x, 9.0 at 4x) due to a fundamental architectural mismatch between training and inference.**

## 1. How the Model is Trained

### 1.1 Training Data Pipeline (`ml/datasets/tfdata.py`)

**Process:**
1. **Deck Encoding**: Each deck is encoded as a sequence: `[BOS, card1, card2, ..., card50, EOS, PAD, ...]`
2. **Teacher Forcing Setup**:
   - `decoder_input = [BOS, card1, card2, ..., card49, PAD]` (shifted by 1)
   - `decoder_target = [card1, card2, ..., card50, EOS, PAD]` (next token prediction)
3. **Deck Shuffling**: 5% of decks are shuffled (reduced from 20%), meaning 95% of decks keep their original order
4. **Frequency Histogram Target** (`freq_hist`):
   ```python
   # Computed from the FULL target deck sequence
   token_one_hot = tf.one_hot(decoder_target, depth=vocab_size)  # (seq_len, vocab_size)
   freq_hist = tf.reduce_sum(token_one_hot, axis=0)  # Sum across sequence -> (vocab_size,)
   freq_hist = freq_hist * special_token_mask  # Zero out special tokens
   freq_hist = tf.minimum(freq_hist, max_copies)  # Cap at 4
   freq_hist = freq_hist / max_copies  # Normalize to [0, 1]
   ```
   - **Key Point**: `freq_hist` is computed from the **complete target deck** (ground truth)
   - Represents: "How many copies of each card should be in the final deck?"

### 1.2 Model Architecture (`ml/models/transformer.py`)

**Outputs:**
1. **Main Output**: `(batch, seq_len, vocab_size)` - Next token logits at each position
2. **Type Auxiliary**: `(batch, 4)` - Type distribution (CHARACTER, EVENT, STAGE, OTHER)
3. **Cost Auxiliary**: `(batch, 11)` - Cost distribution
4. **Frequency Histogram**: `(batch, vocab_size)` - **Global prediction** of card frequencies
   ```python
   pooled_decoder = GlobalAveragePooling1D()(decoder_output)  # (batch, embedding_dim)
   freq_hist_output = Dense(vocab_size, activation="sigmoid")(pooled_decoder)  # (batch, vocab_size)
   ```
   - **Key Point**: `freq_hist` is predicted from **pooled decoder output** (average of all positions)
   - This is a **global prediction** of the entire deck's frequency distribution

### 1.3 Loss Functions (`ml/training/losses.py`)

**Main Loss**: `masked_sparse_categorical_crossentropy`
- Predicts next token given previous tokens (teacher forcing)
- **Weight: 1.0** (dominant)

**Frequency Histogram Loss**: `anti_singleton_loss`
- **Components**:
  1. **MSE Loss** (weight: 0.1): Compares predicted vs actual normalized frequencies
  2. **Entropy Penalty** (weight: 2.0): Penalizes high entropy (encourages concentration)
  3. **Low-Probability Penalty** (weight: 10.0): Penalizes cards with prob < 0.3 (likely 1x)
- **Weight: 100.0** (very high, but still secondary to main loss)

**Problem**: The main loss dominates because:
- It's computed at **every position** (50 positions × batch_size)
- The freq_hist loss is computed **once per sample** (1 × batch_size)
- Even with weight 100.0, the gradient signal from main loss is 50× stronger

### 1.4 Training Process (`ml/training/train.py`)

**What the Model Learns:**
1. **From Main Loss (Teacher Forcing)**:
   - Given `[BOS, card1, card2, ...]`, predict `card3`
   - The model sees: "I've generated card1, card2, so I should predict a different card"
   - **Learns**: "Diversity is rewarded" (each prediction is for a different card)

2. **From Freq_Hist Loss**:
   - Given the **complete decoder sequence**, predict the frequency histogram
   - **Problem**: During training, the decoder sees the **complete ground truth sequence**
   - During inference, the decoder only sees the **partial generated sequence**
   - **Mismatch**: Model learns to predict freq_hist from complete decks, but must predict from partial sequences

## 2. How the Model Generates Decks

### 2.1 Generation Process (`ml/inference/generate.py`)

**Greedy Generation:**
1. Start with `[BOS, leader]`
2. For each step:
   - Pass current sequence to model
   - Get logits for next token
   - Extract `freq_hist` output (predicted from **partial sequence**)
   - Apply inference-time biasing (boosts duplicates, penalizes unseen)
   - Sample next token
   - Repeat until EOS or max length

**The Critical Issue:**
- **During Training**: `freq_hist` is predicted from **complete ground truth sequences**
- **During Inference**: `freq_hist` is predicted from **partial generated sequences**
- **Result**: The `freq_hist` prediction is inaccurate during generation because the model never learned to predict it from partial sequences

### 2.2 Why Inference-Time Biasing Fails

**Even with massive boosts (300+ logits):**
1. **Model's Learned Distribution**: The model has learned such a strong preference for diversity that even 300+ logit boosts cannot overcome it
2. **Freq_Hist Inaccuracy**: The `freq_hist` output during generation is based on partial sequences, so it's not reliable
3. **Main Loss Dominance**: The model was primarily trained to predict diverse next tokens, not realistic frequencies

## 3. Root Cause Analysis

### 3.1 The Fundamental Mismatch

**Training:**
```
Input: [BOS, card1, card2, ..., card50, EOS]  (complete deck)
freq_hist target: computed from complete deck
freq_hist prediction: from pooled decoder (sees complete sequence)
```

**Inference:**
```
Input: [BOS, leader, card1, card2, ...]  (partial deck, growing)
freq_hist prediction: from pooled decoder (sees only partial sequence)
```

**Problem**: The model learns to predict `freq_hist` from complete decks, but must predict from partial decks during generation.

### 3.2 Why Teacher Forcing Encourages Diversity

**Teacher Forcing Pattern:**
- Step 1: `[BOS]` → predict `card1`
- Step 2: `[BOS, card1]` → predict `card2` (different from card1)
- Step 3: `[BOS, card1, card2]` → predict `card3` (different from card1, card2)
- ...

**What the Model Learns:**
- "Given diverse tokens so far, predict another diverse token"
- The model never sees: "Given card1, predict card1 again" (because teacher forcing always provides the next unique card)

**Even with 5% shuffling:**
- 95% of training examples still follow the pattern: "diverse token → diverse token"
- The model learns this pattern very strongly

### 3.3 Why Freq_Hist Loss Doesn't Help Enough

**Gradient Signal Comparison:**
- **Main Loss**: 50 positions × batch_size = 50× gradient signal
- **Freq_Hist Loss**: 1 position × batch_size = 1× gradient signal
- **Even with weight 100.0**: Main loss still dominates because it's computed 50× more often

**Architectural Issue:**
- `freq_hist` is predicted from **pooled decoder output** (global average)
- During training, the pooled output sees the complete sequence
- During inference, the pooled output sees only the partial sequence
- The model cannot accurately predict `freq_hist` from partial sequences because it was trained on complete sequences

## 4. Why Retraining Doesn't Fix It

**Current Training Parameters:**
- `freq_hist_weight = 100.0` (very high)
- `entropy_penalty = 2.0`
- `low_prob_penalty = 10.0`
- `deck_shuffle_rate = 0.05` (low)

**Why It Still Fails:**
1. **Main Loss Still Dominates**: Even with weight 100.0, the main loss has 50× more gradient signal
2. **Architectural Mismatch**: `freq_hist` prediction from partial sequences is inherently inaccurate
3. **Teacher Forcing Bias**: The model still learns "diverse token → diverse token" pattern from 95% of examples

## 5. Solutions (Ranked by Effectiveness)

### 5.1 **Solution 1: Change Training Objective (RECOMMENDED)**

**Problem**: Teacher forcing teaches diversity, not realistic frequencies.

**Solution**: Add a **sequence-level loss** that directly penalizes singleton-heavy generated sequences.

**Implementation:**
1. During training, generate a full sequence (using teacher forcing for efficiency)
2. Compute the actual frequency histogram of the generated sequence
3. Compare it to the target frequency histogram
4. Add this as an additional loss term

**Code Changes:**
```python
# In training loop, after forward pass:
generated_sequence = model.generate(...)  # Full sequence
generated_freq_hist = compute_freq_hist(generated_sequence)
sequence_level_loss = mse_loss(target_freq_hist, generated_freq_hist)
total_loss = main_loss + 100.0 * freq_hist_loss + 50.0 * sequence_level_loss
```

**Pros**: Directly addresses the root cause
**Cons**: Requires generating full sequences during training (slower)

### 5.2 **Solution 2: Curriculum Learning**

**Problem**: Model learns diversity from complete sequences.

**Solution**: Train in stages:
1. **Stage 1**: Train on decks with many duplicates (4x staples)
2. **Stage 2**: Gradually introduce diverse decks
3. **Stage 3**: Fine-tune on full dataset

**Pros**: Helps model learn duplicate patterns first
**Cons**: Requires dataset filtering and multiple training stages

### 5.3 **Solution 3: Change Model Architecture**

**Problem**: `freq_hist` is predicted from pooled decoder (sees complete sequence during training, partial during inference).

**Solution**: Make `freq_hist` prediction **position-aware**:
- Instead of global pooling, use a learned aggregation that works on partial sequences
- Or: Use a separate "frequency predictor" that takes the current sequence and predicts remaining frequencies

**Pros**: Addresses architectural mismatch
**Cons**: Requires significant architecture changes

### 5.4 **Solution 4: Post-Processing (NOT RECOMMENDED)**

**Problem**: Model generates diverse decks.

**Solution**: After generation, replace singleton cards with duplicates of high-frequency cards.

**Pros**: Quick fix
**Cons**: Doesn't fix the model, just patches the output

## 6. Recommended Action Plan

### Immediate (Before Next Training):

1. **Verify Training Data**:
   ```bash
   python scripts/analyze_tournament_duplicates.py
   ```
   - Confirm tournament decks have realistic duplicates (4x staples, 2x tech)
   - Verify merged dataset includes tournament decks

2. **Add Sequence-Level Loss** (Solution 1):
   - Modify training loop to generate full sequences
   - Add loss comparing generated vs target frequency histograms
   - This directly penalizes singleton-heavy generations

3. **Increase Main Loss Weight for Duplicates**:
   - Modify main loss to give higher weight to predicting duplicate tokens
   - When the target is a duplicate (card appears 2nd, 3rd, 4th time), increase loss weight

### Alternative: Accept Inference-Time Post-Processing

If retraining is too expensive/time-consuming:
- Generate decks as-is (diverse)
- Post-process to replace singletons with duplicates
- This is a pragmatic solution, though not ideal

## 7. Summary

**The Core Problem:**
- Model learns "diversity" from teacher forcing (95% of examples)
- `freq_hist` loss is computed on complete sequences during training, but partial sequences during inference
- Main loss dominates (50× more gradient signal)

**Why Current Approach Fails:**
- Increasing `freq_hist_weight` to 100.0 helps, but main loss still dominates
- Reducing `deck_shuffle_rate` to 5% helps, but 95% of examples still teach diversity
- Inference-time biasing cannot overcome the model's learned preference

**The Solution:**
- Add sequence-level loss that directly penalizes singleton-heavy generated sequences
- Or: Accept post-processing as a pragmatic solution

**Bottom Line:**
The model needs to learn from **generated sequences**, not just **ground truth sequences**. The current training only teaches the model to predict next tokens from ground truth, not to generate realistic frequency distributions.

