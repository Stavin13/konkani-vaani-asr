# Parallel Training Setup Guide

## Overview
Train 3 models simultaneously on different Kaggle accounts to save time:
1. **ASR Model** (Main account) - Already running! ✅
2. **Emotion Detection** (Account 2) - Ready to start
3. **Translation Model** (Account 3) - Ready to start

---

## Current Status

### ✅ ASR Training (RUNNING)
- **Account:** Main Kaggle account
- **Status:** Training in progress (Epoch 1 complete)
- **Time:** ~8.5 hours total (50 epochs)
- **GPU:** Dual T4 x2
- **Action:** Monitor progress, download checkpoints when done

---

## Emotion Detection Setup

### Step 1: Generate Training Data (Run Locally or Kaggle)
**Notebook:** `GENERATE_EMOTION_DATA.ipynb`

**What it does:**
- Uses pre-trained multilingual emotion model
- Labels your Konkani text with emotions (joy, sadness, anger, fear, etc.)
- Filters high-confidence predictions
- Balances dataset across emotion classes
- Saves: `data/generated/konkani_emotion_training_data.csv`

**Run this:**
```bash
# Option 1: Run locally
jupyter notebook GENERATE_EMOTION_DATA.ipynb

# Option 2: Upload to Kaggle and run there
```

**Time:** ~10-20 minutes depending on data size

### Step 2: Upload Data to Kaggle
1. Create new Kaggle dataset: "Konkani Emotion Data"
2. Upload: `konkani_emotion_training_data.csv`
3. Make it public or add to your account

### Step 3: Train Custom Model (Kaggle Account 2)
**Notebook:** `KAGGLE_EMOTION_TRAINING.ipynb`

**Setup:**
1. Create new Kaggle notebook
2. Copy content from `KAGGLE_EMOTION_TRAINING.ipynb`
3. Add your emotion dataset to the notebook
4. Update `DATASET_PATH` in Step 2
5. Enable GPU (P100 recommended)
6. Run all cells

**Training time:** ~1 hour on P100

**Output:** Custom Konkani emotion classifier
- Download from `/kaggle/working/konkani_emotion_model/`

---

## Translation Setup

### Step 1: Generate Training Data (Run Locally or Kaggle)
**Notebook:** `GENERATE_TRANSLATION_DATA.ipynb`

**What it does:**
- Uses pre-trained multilingual translation model
- Translates Konkani text to English
- Optionally creates reverse translations (English → Konkani)
- Filters poor quality translations
- Saves: `data/generated/konkani_english_translation_pairs.json`

**Run this:**
```bash
# Option 1: Run locally
jupyter notebook GENERATE_TRANSLATION_DATA.ipynb

# Option 2: Upload to Kaggle and run there
```

**Time:** ~20-30 minutes depending on data size

### Step 2: Upload Data to Kaggle
1. Create new Kaggle dataset: "Konkani English Translation"
2. Upload: `konkani_english_translation_pairs.json`
3. Make it public or add to your account

### Step 3: Train Custom Model (Kaggle Account 3)
**Notebook:** `KAGGLE_TRANSLATION_TRAINING.ipynb`

**Setup:**
1. Create new Kaggle notebook
2. Copy content from `KAGGLE_TRANSLATION_TRAINING.ipynb`
3. Add your translation dataset to the notebook
4. Update `DATASET_PATH` in Step 2
5. Enable GPU (P100 recommended)
6. Run all cells

**Training time:** ~2-3 hours on P100

**Output:** Custom Konkani-English translation model
- Download from `/kaggle/working/konkani_translation_model/`

---

## Quick Start Commands

### 1. Generate Emotion Data
```bash
# Run locally
cd /path/to/project
jupyter notebook GENERATE_EMOTION_DATA.ipynb
# Run all cells, wait for completion
```

### 2. Generate Translation Data
```bash
# Run locally
cd /path/to/project
jupyter notebook GENERATE_TRANSLATION_DATA.ipynb
# Run all cells, wait for completion
```

### 3. Upload to Kaggle
```bash
# Emotion data
kaggle datasets create -p data/generated -r zip

# Translation data
kaggle datasets create -p data/generated -r zip
```

### 4. Start Training on Kaggle
- Open Kaggle notebooks
- Copy training notebook content
- Add datasets
- Enable GPU
- Run all cells
- Turn off internet to save quota

---

## Timeline

| Task | Time | Status |
|------|------|--------|
| ASR Training | 8.5 hours | ✅ Running |
| Generate Emotion Data | 10-20 min | ⏳ Ready |
| Train Emotion Model | 1 hour | ⏳ Ready |
| Generate Translation Data | 20-30 min | ⏳ Ready |
| Train Translation Model | 2-3 hours | ⏳ Ready |

**Total parallel time:** ~8.5 hours (all running simultaneously!)
**vs Sequential:** ~12+ hours

---

## Files Reference

### Data Generation (Run First)
- `GENERATE_EMOTION_DATA.ipynb` - Create emotion labels
- `GENERATE_TRANSLATION_DATA.ipynb` - Create translation pairs

### Training (Run on Kaggle)
- `KAGGLE_EMOTION_TRAINING.ipynb` - Train emotion classifier
- `KAGGLE_TRANSLATION_TRAINING.ipynb` - Train translation model
- `KAGGLE_TRAINING.ipynb` - ASR training (already running)

### Outputs
- `data/generated/konkani_emotion_training_data.csv`
- `data/generated/konkani_english_translation_pairs.json`

---

## Important Notes

1. **Pre-trained Models First:** Always use pre-trained models to generate data, then train custom models on that data

2. **Turn Off Internet:** After installing dependencies on Kaggle, turn off internet to save quota

3. **GPU Selection:** 
   - P100 is best for emotion/translation (faster, more memory)
   - T4 x2 works for ASR (already using)

4. **Parallel Accounts:** Use different Kaggle accounts to run all 3 trainings simultaneously

5. **Download Models:** After training completes, download models from Output tab before notebook expires

---

## Next Steps

1. ✅ ASR is training - just monitor
2. 🔄 Run `GENERATE_EMOTION_DATA.ipynb` locally
3. 🔄 Run `GENERATE_TRANSLATION_DATA.ipynb` locally
4. 📤 Upload generated data to Kaggle
5. 🚀 Start emotion training on Account 2
6. 🚀 Start translation training on Account 3
7. ⏰ Wait ~8.5 hours
8. 📥 Download all trained models!

---

## Questions?

- Check individual notebook comments for detailed steps
- Each notebook has clear step-by-step instructions
- All notebooks follow the same pattern: Load → Process → Train → Save
