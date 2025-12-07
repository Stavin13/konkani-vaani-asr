# Konkani NER Training Guide

Complete guide to train custom Named Entity Recognition model for Konkani.

---

## 🎯 What is NER?

**Named Entity Recognition (NER)** identifies and classifies named entities in text:
- **PER** (Person): Names of people
- **ORG** (Organization): Companies, institutions
- **LOC** (Location): Cities, countries, places
- **MISC** (Miscellaneous): Other entities

**Example:**
```
Input:  "मी मुंबईंत गूगलांत काम करतां"
Output: {
  'locations': ['मुंबई'],
  'organizations': ['गूगल']
}
```

---

## 📦 Requirements

Install dependencies:
```bash
pip install torch transformers pytorch-crf tqdm
```

**Note:** If `pytorch-crf` fails to install, the code will automatically use a simpler model without CRF.

---

## 🚀 Quick Start (2 Commands)

### **Step 1: Auto-label data (10-15 minutes)**

```bash
python3 scripts/auto_label_ner.py \
    --input transcripts_konkani_cleaned.json \
    --output data/ner_labeled_data.json
```

This uses a pre-trained multilingual NER model to automatically label your Konkani transcripts with BIO tags.

**Output:**
- `data/ner_labeled_data.json` - Labeled training data
- `data/ner_labeled_data_label_map.json` - Label mappings

---

### **Step 2: Train custom model (2-3 hours on GPU)**

```bash
python3 train_konkani_ner.py \
    --data_file data/ner_labeled_data.json \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --device cuda \
    --checkpoint_dir checkpoints/ner
```

**For Mac (MPS):**
```bash
python3 train_konkani_ner.py \
    --data_file data/ner_labeled_data.json \
    --batch_size 16 \
    --num_epochs 20 \
    --device mps \
    --checkpoint_dir checkpoints/ner
```

**Output:**
- `checkpoints/ner/best_ner_model.pt` - Best model
- `checkpoints/ner/vocabularies.json` - Word/char vocabularies
- Checkpoints saved every 5 epochs

---

## 📊 Expected Results

**Dataset:**
- ~2,500 labeled samples
- ~50,000 tokens
- 9 NER tags (B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O)

**Model:**
- Architecture: BiLSTM-CRF
- Parameters: ~2-3M
- Training time: 2-3 hours on GPU
- Expected F1: 75-80%

---

## 🔧 Advanced Options

### Test on small sample first:

```bash
# Auto-label only 100 samples for testing
python3 scripts/auto_label_ner.py \
    --input transcripts_konkani_cleaned.json \
    --output data/ner_test_data.json \
    --max_samples 100

# Train on small dataset
python3 train_konkani_ner.py \
    --data_file data/ner_test_data.json \
    --num_epochs 5 \
    --device cuda
```

### Use CRF layer (better accuracy):

```bash
python3 train_konkani_ner.py \
    --data_file data/ner_labeled_data.json \
    --use_crf \
    --device cuda
```

**Note:** CRF requires `pytorch-crf` package. If not installed, model will use softmax instead.

---

## 📁 File Structure

```
konkani/
├── scripts/
│   └── auto_label_ner.py          # Auto-labeling script
├── models/
│   └── konkani_ner.py             # NER model definition
├── train_konkani_ner.py           # Training script
├── data/
│   ├── ner_labeled_data.json      # Auto-labeled data (generated)
│   └── ner_labeled_data_label_map.json
├── checkpoints/
│   └── ner/
│       ├── best_ner_model.pt      # Best model (generated)
│       ├── vocabularies.json      # Vocabularies (generated)
│       └── ner_checkpoint_epoch_*.pt
└── NER_TRAINING_GUIDE.md          # This file
```

---

## 🎓 How It Works

### **Step 1: Auto-labeling**

1. Load pre-trained XLM-RoBERTa NER model (trained on 10+ languages)
2. Process each Konkani transcript
3. Extract entities (persons, locations, organizations)
4. Convert to BIO format:
   - **B-PER**: Beginning of person name
   - **I-PER**: Inside person name
   - **O**: Outside any entity

**Example:**
```
Text:    "Stavin Fernandes works at Google in Mumbai"
Tokens:  ["Stavin", "Fernandes", "works", "at", "Google", "in", "Mumbai"]
Labels:  ["B-PER", "I-PER",     "O",     "O",  "B-ORG",  "O",  "B-LOC"]
```

### **Step 2: Training**

1. Load auto-labeled data
2. Build word and character vocabularies
3. Train BiLSTM-CRF model:
   - **Embedding layer**: Convert words to vectors
   - **BiLSTM**: Capture context from both directions
   - **CRF**: Ensure valid tag sequences (e.g., I-PER must follow B-PER)
4. Evaluate on validation set
5. Save best model

---

## 🔍 Monitoring Training

Training will show progress like this:

```
Epoch 1/20
Epoch 1: 100%|████████| 64/64 [00:45<00:00,  1.41it/s, loss=2.1234]
Validation: 100%|████████| 16/16 [00:05<00:00,  2.89it/s]

Epoch 1/20
  Train Loss: 2.1234
  Val Loss: 1.8765
  Val F1: 0.6543

✅ Saved best model with F1: 0.6543
```

**What to look for:**
- Loss should decrease over epochs
- F1 score should increase (target: 0.75-0.80)
- Best model saved when F1 improves

---

## 🧪 Testing the Model

After training, test on a sample:

```python
import torch
import json
from models.konkani_ner import create_ner_model

# Load model
checkpoint = torch.load('checkpoints/ner/best_ner_model.pt')
model = create_ner_model(vocab_size=5000, num_tags=9)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load vocabularies
with open('checkpoints/ner/vocabularies.json') as f:
    vocabs = json.load(f)
    word2id = vocabs['word2id']

# Test text
text = "मी मुंबईंत गूगलांत काम करतां"
tokens = text.split()

# Convert to IDs
word_ids = torch.tensor([[word2id.get(t, 1) for t in tokens]])

# Predict
with torch.no_grad():
    predictions = model(word_ids)

print(f"Text: {text}")
print(f"Entities: {predictions}")
```

---

## ⏱️ Timeline

**For your project (Sunday evening):**

| Time | Task | Duration |
|------|------|----------|
| 8:00 PM | Install dependencies | 5 min |
| 8:05 PM | Run auto-labeling | 15 min |
| 8:20 PM | Start NER training | 2-3 hours |
| 11:00 PM | Training complete | - |

**Parallel with ASR:**
- ASR training runs on Colab (GPU)
- NER training runs on your Mac (MPS) or second Colab session
- Both finish by Sunday night!

---

## 🐛 Troubleshooting

### "pytorch-crf not found"
```bash
pip install pytorch-crf
# OR just run without --use_crf flag
```

### "Out of memory"
```bash
# Reduce batch size
python3 train_konkani_ner.py --batch_size 16  # or 8
```

### "Transformers model download slow"
```bash
# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache
```

### "Auto-labeling taking too long"
```bash
# Test on small sample first
python3 scripts/auto_label_ner.py --max_samples 100
```

---

## ✅ Integration with Complete System

After training, integrate into your audio analyzer:

```python
class KonkaniAudioAnalyzer:
    def __init__(self):
        self.asr_model = load_asr_model()
        self.sentiment_model = load_sentiment_model()
        self.emotion_model = load_emotion_model()
        self.ner_model = load_ner_model()  # ← NEW
        self.translator = load_translator()
    
    def analyze(self, audio_file):
        # ASR
        text = self.asr_model.transcribe(audio_file)
        
        # NER ← NEW
        entities = self.ner_model.extract_entities(text)
        
        # Other models...
        
        return {
            'transcript': text,
            'entities': entities,  # ← NEW
            'sentiment': sentiment,
            'emotion': emotion,
            'translation': english_text
        }
```

---

## 📚 References

- **XLM-RoBERTa NER**: https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl
- **BiLSTM-CRF**: https://arxiv.org/abs/1508.01991
- **BIO Tagging**: https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)

---

**Ready to start? Run the commands above!** 🚀
