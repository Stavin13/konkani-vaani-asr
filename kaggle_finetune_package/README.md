# KonkaniVani ASR Fine-tuning Dataset

## 📊 Dataset Contents

### Model
- `best_model.pt` - Best checkpoint (val_loss: 2.0637, vocab_size: 81)

### Mega Dataset
- `data/konkani-mega-dataset/manifests/train.json` - 64,106 training samples
- `data/konkani-mega-dataset/manifests/val.json` - 8,013 validation samples  
- `data/konkani-mega-dataset/vocab.json` - 192-character vocabulary

### Audio Samples
- `audio_samples/` - Sample audio files for testing

## 🚀 Usage

1. Upload this dataset to Kaggle
2. Use the notebook: `KAGGLE_FINETUNE_BEST_MODEL_MEGA_DATASET.ipynb`
3. Update paths in notebook to match your dataset

## 📝 Notes

- Original model: vocab_size=81, val_loss=2.0637
- Target: Improve with 80K+ samples, vocab_size=192
- Expected: val_loss < 1.8

## 🔗 Audio Files

**IMPORTANT**: You need to upload the full audio dataset separately.
The manifest files reference audio paths that need to be available.

Audio files are located at:
```
/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/Data/
```

Create a separate Kaggle dataset with these audio files.
