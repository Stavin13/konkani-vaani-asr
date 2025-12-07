# Konkani NER Model - Usage Guide

## Overview

The Konkani Named Entity Recognition (NER) model identifies entities in Konkani text:
- **PER**: Person names
- **ORG**: Organizations
- **LOC**: Locations
- **MISC**: Miscellaneous entities

## Model Architecture

**BiLSTM-CRF** with:
- Word embeddings (128 dim)
- Character-level CNN features (32 dim)
- Bidirectional LSTM (256 hidden dim, 2 layers)
- CRF decoder for sequence labeling

## Files Structure

```
data/
├── ner_labeled_data.json              # Training data (5.8MB)
└── ner_labeled_data_label_map.json    # Label mappings

checkpoints/ner/
├── best_ner_model.pt                  # Best model checkpoint
├── ner_checkpoint_epoch_*.pt          # Training checkpoints
└── vocabularies.json                  # Word & char vocabularies

models/
└── konkani_ner.py                     # Model architecture

scripts/
├── auto_label_ner.py                  # Auto-labeling script
├── test_ner_model.py                  # Testing script
└── demo_ner.py                        # Demo with samples
```

## Quick Start

### 1. Test the Model

```bash
# Run demo with labeled samples
python scripts/demo_ner.py

# Test with custom text
python scripts/test_ner_model.py --text "तुमचे नाव काय आहे"
```

### 2. Use in Your Code

```python
import torch
import json
from models.konkani_ner import KonkaniNER

# Load vocabularies
with open('checkpoints/ner/vocabularies.json', 'r') as f:
    vocabs = json.load(f)

word2id = vocabs['word2id']
char2id = vocabs['char2id']

# Load label mapping
with open('data/ner_labeled_data_label_map.json', 'r') as f:
    label_data = json.load(f)

id2label = label_data['id2label']

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = KonkaniNER(
    vocab_size=len(word2id),
    char_vocab_size=len(char2id),
    num_tags=len(id2label),
    embedding_dim=128,
    char_embedding_dim=32,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3
)

# Load checkpoint
checkpoint = torch.load('checkpoints/ner/best_ner_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Make predictions
# (See scripts/test_ner_model.py for full prediction code)
```

## Model Statistics

- **Vocabulary**: 31,199 words
- **Character vocab**: 196 characters
- **Training epoch**: 18
- **Entity types**: 4 (PER, ORG, LOC, MISC)
- **BIO tagging**: 9 labels (B-/I- for each type + O)

## Example Predictions

**Sample 1:**
```
Text: मझ्या वाताराण न्योजिक स्तान खाय असा माझा वाारां कलंगुट चरच असा

Entities:
  [LOC] कलंगुट
  [LOC] चर्च
```

**Sample 2:**
```
Text: ना आलली कित्या ओडखण आलली आणि टi लबत शकोन

Entities:
  [PER] आणिशकोन
  [PER] शकन
```

## Auto-Labeling New Data

To label new transcripts:

```bash
python scripts/auto_label_ner.py \
    --input transcripts_konkani_cleaned.json \
    --output data/ner_labeled_data_new.json \
    --max_samples 100
```

This uses a pre-trained multilingual NER model (XLM-RoBERTa) to generate initial labels.

## Training

The model was trained for 50 epochs with:
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Dropout: 0.3

Checkpoints saved every 5 epochs in `checkpoints/ner/`.

## Requirements

```bash
pip install torch pytorch-crf transformers
```

## Notes

- Model works best with clean Devanagari text
- Character-level features help with OOV words
- CRF layer ensures valid BIO tag sequences
- Best checkpoint selected based on validation F1 score

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'torchcrf'`
```bash
pip install pytorch-crf
```

**Issue**: Model predicts all 'O' tags
- Check if input text matches training data format
- Ensure vocabularies are loaded correctly
- Verify checkpoint file is not corrupted

## Next Steps

1. Fine-tune on domain-specific data
2. Add more entity types if needed
3. Integrate with ASR pipeline for end-to-end processing
4. Export to ONNX for faster inference
