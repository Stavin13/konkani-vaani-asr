# KonkaniVani ASR Quick Test Package

## 📊 Contents

### Model
- `best_model.pt` - Best checkpoint (val_loss: 2.0637, vocab_size: 81)

### Dataset Metadata
- `data/konkani-mega-dataset/manifests/train.json` - 64,106 training samples
- `data/konkani-mega-dataset/manifests/val.json` - 8,013 validation samples  
- `data/konkani-mega-dataset/vocab.json` - 81-character vocabulary

## ⚠️ Audio Files Not Included

This package contains only metadata for quick testing. 
Audio files need to be uploaded separately or paths updated in the notebook.

## 🚀 Quick Start

1. Upload this dataset to Kaggle
2. Use notebook: `KAGGLE_FINETUNE_BEST_MODEL_MEGA_DATASET.ipynb`
3. Update audio paths or use subset for testing

## 📝 Next Steps

For full training:
1. Upload audio files separately
2. Update manifest paths in notebook
3. Or start with subset testing
