# 🏗️ Custom Konkani Models Architecture

## Overview: All Custom-Built Models

All three models are **built from scratch** specifically for Konkani, not fine-tuned from pre-trained models.

---

## 1. ✅ ASR Model (Already Trained)

### Architecture: Custom Conformer-Transformer
**File:** `models/konkanivani_asr.py`

```
Input: Audio (mel-spectrograms, 80 features)
  ↓
Conformer Encoder (12 layers)
├── Convolution Subsampling
├── Multi-Head Self-Attention
├── Convolution Module
└── Feed-Forward Network
  ↓
Transformer Decoder (6 layers)
├── Masked Self-Attention
├── Cross-Attention (to encoder)
└── Feed-Forward Network
  ↓
Dual Output Heads:
├── CTC Head → Character probabilities
└── Attention Head → Character sequence
  ↓
Output: Konkani Text
```

**Specifications:**
- **Parameters:** 24.7M
- **d_model:** 256
- **Encoder layers:** 12 (Conformer)
- **Decoder layers:** 6 (Transformer)
- **Heads:** 4
- **Loss:** Hybrid CTC (30%) + Attention (70%)
- **Training:** 16 epochs on 2x Tesla T4
- **Status:** ✅ Trained (val_loss: 9.47)

---

## 2. 🆕 Translation Model (Custom)

### Architecture: Custom Seq2Seq Transformer
**File:** `models/konkani_custom_translator.py`

```
Input: Konkani Text (tokenized)
  ↓
Source Embedding + Positional Encoding
  ↓
Transformer Encoder (6 layers)
├── Multi-Head Self-Attention
├── Feed-Forward Network
└── Layer Normalization
  ↓
Transformer Decoder (6 layers)
├── Masked Self-Attention
├── Cross-Attention (to encoder)
├── Feed-Forward Network
└── Layer Normalization
  ↓
Output Projection → English Vocabulary
  ↓
Output: English Text
```

**Specifications:**
- **Parameters:** 17.5M
- **d_model:** 256
- **Encoder layers:** 6
- **Decoder layers:** 6
- **Heads:** 8
- **Feedforward dim:** 1024
- **Dropout:** 0.1
- **Max length:** 512 tokens
- **Loss:** Cross-Entropy with Label Smoothing
- **Status:** 🔄 Ready to train

**Key Features:**
- ✅ Greedy decoding for inference
- ✅ Beam search capable
- ✅ Attention visualization
- ✅ Teacher forcing during training

---

## 3. 🆕 Emotion Model (Custom)

### Architecture: BiLSTM + Attention
**File:** `models/konkani_custom_emotion.py`

```
Input: Konkani Text (tokenized)
  ↓
Embedding Layer (128 dim)
  ↓
Bidirectional LSTM (2 layers, 256 hidden)
├── Forward LSTM
└── Backward LSTM
  ↓
Attention Mechanism
├── Calculate attention scores
├── Apply softmax
└── Weighted sum of LSTM outputs
  ↓
Layer Normalization
  ↓
Feed-Forward Layers
├── FC1 (512 → 256) + ReLU
└── FC2 (256 → 7 emotions)
  ↓
Output: Emotion Probabilities
```

**Specifications:**
- **Parameters:** 3.1M
- **Embedding dim:** 128
- **Hidden dim:** 256 (512 bidirectional)
- **LSTM layers:** 2
- **Dropout:** 0.3
- **Emotions:** 7 classes
  - anger, disgust, fear, joy, neutral, sadness, surprise
- **Loss:** Cross-Entropy with Label Smoothing (0.1)
- **Status:** 🔄 Ready to train

**Key Features:**
- ✅ Attention weights visualization
- ✅ Bidirectional context
- ✅ Label smoothing for better generalization
- ✅ Packed sequences for efficiency

---

## 📊 Model Comparison

| Feature | ASR | Translation | Emotion |
|---------|-----|-------------|---------|
| **Architecture** | Conformer-Transformer | Seq2Seq Transformer | BiLSTM-Attention |
| **Parameters** | 24.7M | 17.5M | 3.1M |
| **Input** | Audio (80 mel) | Konkani text | Konkani text |
| **Output** | Konkani text | English text | 7 emotions |
| **Layers** | 12 enc + 6 dec | 6 enc + 6 dec | 2 LSTM + 2 FC |
| **Attention** | Multi-head | Multi-head | Single-head |
| **Training Time** | 6-8 hours | 3-4 hours | 1-2 hours |
| **GPU Memory** | ~12GB | ~8GB | ~4GB |
| **Status** | ✅ Trained | 🔄 To train | 🔄 To train |

---

## 🎯 Why Custom Models?

### Advantages:
1. **Full Control:** Optimize for Konkani language characteristics
2. **Smaller Size:** More efficient than large pre-trained models
3. **Faster Inference:** Optimized for specific tasks
4. **No Dependencies:** Don't rely on external model hubs
5. **Customizable:** Easy to modify for specific needs

### Trade-offs:
1. **More Training Data Needed:** No pre-training benefits
2. **Longer Training:** Start from random initialization
3. **Lower Initial Performance:** Need more epochs to converge

---

## 🔧 Training Configuration

### Translation Model
```python
config = {
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    'max_len': 512
}

training_config = {
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'batch_size': 32,
    'num_epochs': 30,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'grad_clip': 1.0
}
```

### Emotion Model
```python
config = {
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True
}

training_config = {
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 64,
    'num_epochs': 20,
    'label_smoothing': 0.1,
    'grad_clip': 5.0
}
```

---

## 📈 Expected Performance

### Translation Model
- **BLEU Score:** 30-40 (after 30 epochs)
- **Training Time:** 3-4 hours on P100
- **Convergence:** ~15-20 epochs
- **Best Practices:**
  - Use teacher forcing ratio: 0.5
  - Implement beam search (k=5)
  - Add attention regularization

### Emotion Model
- **Accuracy:** 75-85% (after 20 epochs)
- **Training Time:** 1-2 hours on P100
- **Convergence:** ~10-15 epochs
- **Best Practices:**
  - Balance emotion classes
  - Use attention weights for interpretability
  - Add class weights if imbalanced

---

## 🚀 Usage Examples

### Translation Model
```python
from models.konkani_custom_translator import create_custom_translation_model

# Create model
model = create_custom_translation_model(
    src_vocab_size=5000,  # Konkani vocab
    tgt_vocab_size=10000  # English vocab
)

# Translate
konkani_tokens = tokenizer.encode("हांव खूश आसा")
english_tokens = model.translate(konkani_tokens)
english_text = tokenizer.decode(english_tokens)
```

### Emotion Model
```python
from models.konkani_custom_emotion import create_custom_emotion_model

# Create model
model = create_custom_emotion_model(
    vocab_size=5000,
    num_emotions=7
)

# Predict emotion
tokens = tokenizer.encode("हांव खूश आसा")
prediction, probabilities, attention = model.predict(tokens)
emotion = emotion_labels[prediction]
```

---

## 🎓 Model Architecture Decisions

### Why Transformer for Translation?
- ✅ Better at capturing long-range dependencies
- ✅ Parallel processing (faster training)
- ✅ State-of-the-art for translation tasks
- ✅ Attention mechanism for alignment

### Why BiLSTM for Emotion?
- ✅ Smaller and faster than Transformer
- ✅ Good for sequence classification
- ✅ Bidirectional context important for emotions
- ✅ Attention helps interpretability
- ✅ Less data needed to train

### Why Conformer for ASR?
- ✅ Combines CNN and Transformer benefits
- ✅ Best for audio processing
- ✅ Captures both local and global features
- ✅ State-of-the-art for speech recognition

---

## 📦 Model Files

```
models/
├── konkanivani_asr.py              # ✅ ASR (trained)
├── konkani_custom_translator.py    # 🆕 Translation (custom)
├── konkani_custom_emotion.py       # 🆕 Emotion (custom)
├── konkani_translator.py           # (old fine-tuned version)
└── konkani_ner.py                  # NER model
```

---

## 🎯 Next Steps

1. **Generate Training Data** using pre-trained models
2. **Train Translation Model** (~3-4 hours)
3. **Train Emotion Model** (~1-2 hours)
4. **Evaluate Performance** on test sets
5. **Fine-tune** if needed
6. **Deploy** all three models together

---

**All models are custom-built from scratch for Konkani!** 🚀
