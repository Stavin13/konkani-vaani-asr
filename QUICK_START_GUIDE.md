# 🚀 Quick Start: Train Your Custom Models

## Step-by-Step Guide

### Step 1: Generate Training Data (Run Locally or on Kaggle)

#### A. Generate Translation Data
**Use this notebook:** `notebooks/GENERATE_TRANSLATION_DATA.ipynb`

```bash
# Run locally
jupyter notebook notebooks/GENERATE_TRANSLATION_DATA.ipynb
```

**What it does:**
- Uses pre-trained model (Helsinki-NLP/opus-mt-mul-en)
- Generates Konkani→English translation pairs
- Saves to `data/translation/train.json`

**Output:**
```json
[
  {"konkani": "हांव घरा वता", "english": "I am going home"},
  {"konkani": "तुजें नांव कितें?", "english": "What is your name?"}
]
```

#### B. Generate Emotion Data
**Use this notebook:** `notebooks/GENERATE_EMOTION_DATA.ipynb`

```bash
# Run locally
jupyter notebook notebooks/GENERATE_EMOTION_DATA.ipynb
```

**What it does:**
- Uses pre-trained emotion model (j-hartmann/emotion-english-distilroberta-base)
- Labels Konkani text with emotions
- Saves to `data/emotion/train.csv`

**Output:**
```csv
text,emotion
"हांव खूश आसा",joy
"हांव दुखी आसा",sadness
```

---

### Step 2: Train Your Custom Models on Kaggle

#### Option A: Use the Complete Notebook (Easiest!)

**Use:** `notebooks/KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb`

This notebook does EVERYTHING:
1. ✅ Generates translation data
2. ✅ Generates emotion data
3. ✅ Trains custom translation model
4. ✅ Trains custom emotion model

**Steps:**
1. Upload to Kaggle
2. Enable GPU (P100 or T4)
3. Click "Run All"
4. Wait ~5-6 hours
5. Download trained models

#### Option B: Use Separate Notebooks (More Control)

**For Translation:**
- Generate data: `GENERATE_TRANSLATION_DATA.ipynb` (local)
- Train model: Upload data + custom model to Kaggle

**For Emotion:**
- Generate data: `GENERATE_EMOTION_DATA.ipynb` (local)
- Train model: Upload data + custom model to Kaggle

---

## 📁 Files You Need

### For Local Data Generation:
```
notebooks/
├── GENERATE_TRANSLATION_DATA.ipynb  ← Run this first
└── GENERATE_EMOTION_DATA.ipynb      ← Run this second
```

### For Kaggle Training:
```
Upload these to Kaggle:
├── notebooks/KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb  ← Main notebook
├── models/konkani_custom_translator.py                  ← Translation model
└── models/konkani_custom_emotion.py                     ← Emotion model
```

---

## 🎯 Recommended Workflow

### Easiest Way (All on Kaggle):

```bash
# 1. Upload to Kaggle
- KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb
- konkani_custom_translator.py
- konkani_custom_emotion.py

# 2. Run notebook
# It will:
# - Generate data automatically
# - Train both models
# - Save trained models

# 3. Download results
kaggle kernels output YOUR_USERNAME/NOTEBOOK_NAME -p ./outputs
```

### Alternative Way (Generate Data Locally, Train on Kaggle):

```bash
# 1. Generate data locally (faster, no GPU needed)
jupyter notebook notebooks/GENERATE_TRANSLATION_DATA.ipynb
jupyter notebook notebooks/GENERATE_EMOTION_DATA.ipynb

# 2. Upload to Kaggle as datasets
kaggle datasets create -p data/translation
kaggle datasets create -p data/emotion

# 3. Create Kaggle notebook with your custom models
# 4. Add datasets to notebook
# 5. Train models
```

---

## 📊 What Each File Does

| File | Purpose | Where to Run | Time |
|------|---------|--------------|------|
| `GENERATE_TRANSLATION_DATA.ipynb` | Create translation pairs | Local/Kaggle | 30 min |
| `GENERATE_EMOTION_DATA.ipynb` | Label emotions | Local/Kaggle | 30 min |
| `KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb` | Generate data + Train models | Kaggle | 5-6 hours |
| `konkani_custom_translator.py` | Translation model architecture | Upload to Kaggle | - |
| `konkani_custom_emotion.py` | Emotion model architecture | Upload to Kaggle | - |

---

## 🚀 Quick Commands

### Generate Data Locally:
```bash
# Translation data
jupyter notebook notebooks/GENERATE_TRANSLATION_DATA.ipynb

# Emotion data
jupyter notebook notebooks/GENERATE_EMOTION_DATA.ipynb
```

### Upload to Kaggle:
```bash
# Upload notebook
# Go to https://www.kaggle.com/code
# Click "New Notebook" → "Upload"
# Select: KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb

# Upload model files
# In Kaggle notebook, click "Add Data" → "Upload"
# Upload: konkani_custom_translator.py, konkani_custom_emotion.py
```

### Download Trained Models:
```bash
kaggle kernels output YOUR_USERNAME/NOTEBOOK_NAME -p ./trained_models
```

---

## 💡 Recommendation

**Use the complete Kaggle notebook** (`KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb`)

**Why?**
- ✅ Everything in one place
- ✅ No need to upload datasets
- ✅ Generates data automatically
- ✅ Trains both models
- ✅ Just click "Run All"

**Only use local generation if:**
- You have a large Konkani corpus locally
- You want to customize the data generation
- You want to inspect data before training

---

## 🎯 TL;DR - Simplest Path

```bash
1. Go to Kaggle: https://www.kaggle.com/code

2. Upload these 3 files:
   - KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb
   - konkani_custom_translator.py
   - konkani_custom_emotion.py

3. Enable GPU (P100)

4. Click "Run All"

5. Wait 5-6 hours

6. Download trained models from Output tab

Done! 🎉
```

---

## 📞 Need Help?

Check these docs:
- `docs/TRAINING_CUSTOM_MODELS.md` - Detailed training guide
- `docs/CUSTOM_MODELS_ARCHITECTURE.md` - Model architecture details
- `docs/ONE_NOTEBOOK_GUIDE.md` - Notebook usage guide
