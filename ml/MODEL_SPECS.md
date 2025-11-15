# Model Specifications

This document provides specifications for the models in this project and guidance on public presentation.

## Model Overview

This project contains two main models:

1. **Intent Classifier** - Encoder-only transformer for classifying user prompts
2. **Deck Transformer** - Encoder-decoder transformer for generating decks from prompts

## Intent Classifier Model

### Architecture
- **Type**: Encoder-only Transformer
- **Purpose**: Multi-task classification of user prompts into colors, keywords, leaders, and strategies

### Default Configuration
```python
embedding_dim: int = 256
num_encoder_layers: int = 4
num_heads: int = 8
ff_dim: int = 512
dropout_rate: float = 0.1
```

### Input/Output
- **Input**: Prompt tokens (sequence of tokenized text)
- **Outputs**:
  - `colors`: Binary classification for each color (6 classes)
  - `keywords`: Binary classification for each keyword (~1000 classes)
  - `leaders`: Binary classification for each leader (245 classes)
  - `strategies`: Binary classification for each strategy tag (5 classes)

### Architecture Details
- Embedding layer with positional encoding
- 4 Transformer encoder layers with:
  - Multi-head self-attention (8 heads)
  - Feed-forward network (512 dim)
  - Layer normalization and residual connections
- Global average pooling
- Multi-task classification heads with sigmoid activation

### Estimated Parameters
- Embedding: `prompt_vocab_size × 256`
- Encoder layers: ~4 × (attention + FFN) ≈ 4 × 1M = ~4M parameters
- Classification heads: ~256 × (6 + 1000 + 245 + 5) ≈ 320K parameters
- **Total**: Approximately 4-5M parameters (varies with vocabulary size)

## Deck Transformer Model

### Architecture
- **Type**: Encoder-Decoder Transformer
- **Purpose**: Sequence-to-sequence generation of deck lists from prompts

### Default Configuration
```python
embedding_dim: int = 256
ff_dim: int = 512
num_heads: int = 8
num_layers: int = 4  # Both encoder and decoder
dropout_rate: float = 0.1
coverage_loss_weight: float = 0.01
```

### Input/Output
- **Inputs**:
  - Prompt tokens (encoder input)
  - Deck tokens (decoder input, for teacher forcing)
  - Optional card features (cost, power, color, type, abilities)
- **Outputs**:
  - `main`: Card sequence prediction (vocab_size classes)
  - `type_aux`: Card type distribution (4 classes)
  - `cost_aux`: Cost distribution (11 classes)
  - `freq_hist`: Card frequency histogram (vocab_size classes)

### Architecture Details
- **Encoder**:
  - Embedding layer with positional encoding
  - 4 Transformer encoder layers
  - Multi-head self-attention
- **Decoder**:
  - Embedding layer (optionally enhanced with card features)
  - Positional encoding
  - 4 Transformer decoder layers with:
    - Self-attention (with look-ahead mask)
    - Cross-attention to encoder
    - Feed-forward network
  - Coverage loss regularization

### Estimated Parameters
- Embeddings: `(prompt_vocab_size + deck_vocab_size) × 256`
- Encoder: ~4 × 1M = ~4M parameters
- Decoder: ~4 × 1.5M = ~6M parameters (includes cross-attention)
- Output heads: `deck_vocab_size × 256 × 4` ≈ 1-2M parameters
- **Total**: Approximately 10-15M parameters (varies with vocabulary sizes)

## Checking Model Specifications

### Using the Inspection Script

```bash
# For intent classifier
python -m ml.inspect_model_specs \
  --model models/intent_run_YYYYMMDD-HHMMSS/intent_classifier.keras \
  --vocab-dir models/intent_run_YYYYMMDD-HHMMSS/vocab

# For deck transformer
python -m ml.inspect_model_specs \
  --model models/run_YYYYMMDD-HHMMSS/deck_transformer.keras \
  --vocab-dir models/run_YYYYMMDD-HHMMSS/vocab
```

### Manual Inspection

You can also check model specs programmatically:

```python
import tensorflow as tf
from ml.models.transformer import TransformerConfig
from ml.models.intent_classifier import IntentClassifierConfig

# Check default configs
print("Transformer Config:", TransformerConfig())
print("Intent Classifier Config:", IntentClassifierConfig())

# Load and inspect a model
tf.keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model("path/to/model.keras", compile=False, safe_mode=False)
model.summary()  # Print detailed architecture
print(f"Total parameters: {model.count_params():,}")
```

## Public Presentation Assessment

### ✅ Safe to Present Publicly

**Architecture:**
- Custom transformer architecture built from scratch
- Uses standard TensorFlow/Keras components
- No proprietary pre-trained models
- Open-source implementation

**Licensing:**
- Model code is original work
- Uses open-source TensorFlow/Keras (Apache 2.0)
- No third-party model licenses

**Data:**
- Training data appears to be synthetic or from public OPTCG API
- Card data sourced from public API (optcgapi.com)
- Verify data attribution requirements

### ⚠️ Considerations

1. **Training Data Attribution**
   - Ensure training data sources are properly attributed
   - Check if synthetic data generation methods need citation

2. **Card Data/Images**
   - Card images from OPTCG API - verify API terms of service
   - Card data may have copyright restrictions
   - Images are likely copyrighted by Bandai Namco

3. **Model Weights**
   - Model weights themselves are safe to share (trained from scratch)
   - No copyrighted content embedded in weights
   - Weights are mathematical parameters, not copyrighted material

4. **API Usage**
   - Respect OPTCG API rate limits and terms
   - Attribute API usage appropriately

### Recommended Presentation Format

When presenting this model publicly, include:

1. **Architecture Description**
   - Model type (transformer encoder/decoder)
   - Layer counts and dimensions
   - Parameter counts

2. **Training Details**
   - Dataset size and composition
   - Training hyperparameters
   - Evaluation metrics

3. **Attributions**
   - TensorFlow/Keras framework
   - OPTCG API for card data
   - Any research papers that inspired the architecture

4. **Limitations**
   - Model performance characteristics
   - Known limitations or biases

## Example Model Spec Summary

**Intent Classifier (Example Run)**
- **File Size**: ~111 MB
- **Parameters**: ~4-5M
- **Architecture**: 4-layer encoder-only transformer
- **Embedding Dim**: 256
- **Attention Heads**: 8
- **Vocabulary Sizes**:
  - Prompt: ~10,000 tokens
  - Colors: 6
  - Keywords: ~1,000
  - Leaders: 245
  - Strategies: 5

## Quick Reference

### Get Model File Size
```bash
# Windows
dir models\*\*.keras /s

# Linux/Mac
find models -name "*.keras" -exec ls -lh {} \;
```

### Count Parameters (Python)
```python
import tensorflow as tf
tf.keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model("model.keras", compile=False, safe_mode=False)
print(f"Parameters: {model.count_params():,}")
```

### View Model Architecture
```python
model.summary()  # Detailed layer-by-layer breakdown
```

