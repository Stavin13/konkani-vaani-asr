# Kaggle Parallel Training Guide
## Train ASR, Translation & Emotion Detection Simultaneously

## 🎯 Overview

You can train all three models at the same time on Kaggle using multiple notebooks:

| Model | Notebook | GPU | Time | Priority |
|-------|----------|-----|------|----------|
| **ASR** | KAGGLE_TRAINING_OPTIMIZED.ipynb | 2x T4 | ~6-8h | ⭐⭐⭐ High |
| **Translation** | KAGGLE_TRANSLATION_TRAINING.ipynb | 1x P100 | ~2-3h | ⭐⭐ Medium |
| **Emotion** | KAGGLE_EMOTION_TRAINING.ipynb | 1x P100 | ~1-2h | ⭐ Low |

## 📊 Strategy: Parallel Training

### Option 1: Single Account (Sequential)
Run one at a time within your 30-hour weekly quota:

```
Day 1: ASR (8 hours) → 22 hours left
Day 2: Translation (3 hours) → 19 hours left  
Day 3: Emotion (2 hours) → 17 hours left
```

### Option 2: Multiple Accounts (Parallel) ⚡
Use 3 different Kaggle accounts to run all simultaneously:

```
Account A: ASR training (8 hours)
Account B: Translation training (3 hours)  
Account C: Emotion training (2 hours)

Total wall time: 8 hours (instead of 13!)
```

## 🚀 Quick Start

### 1. ASR Training (Already Done! ✅)
You've already completed this with `KAGGLE_TRAINING_OPTIMIZED.ipynb`

### 2. Translation Training

**Prepare Dataset:**
```bash
# Create translation dataset
python scripts/prepare_translation_data.py

# Package for Kaggle
zip -r konkani_translation_data.zip data/translation/
```

**Upload to Kaggle:**
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload `konkani_translation_data.zip`
4. Name it: "konkani-english-translation"

**Create Notebook:**
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload `notebooks/KAGGLE_TRANSLATION_TRAINING.ipynb`
4. Add your translation dataset
5. Enable GPU (P100 or T4)
6. Run all cells

### 3. Emotion Detection Training

**Prepare Dataset:**
```bash
# Generate emotion data
jupyter notebook notebooks/GENERATE_EMOTION_DATA.ipynb

# Package for Kaggle
zip -r konkani_emotion_data.zip data/emotion/
```

**Upload to Kaggle:**
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload `konkani_emotion_data.zip`
4. Name it: "konkani-emotion-detection"

**Create Notebook:**
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload `notebooks/KAGGLE_EMOTION_TRAINING.ipynb`
4. Add your emotion dataset
5. Enable GPU (P100 or T4)
6. Run all cells

## 📦 What You Need to Upload

### For Translation:
```
konkani-english-translation/
├── train.json          # Konkani-English pairs
├── val.json           # Validation pairs
└── test.json          # Test pairs
```

Format:
```json
[
  {"konkani": "हांव घरा वता", "english": "I am going home"},
  {"konkani": "तुजें नांव कितें?", "english": "What is your name?"}
]
```

### For Emotion:
```
konkani-emotion-detection/
├── train.csv          # Text + emotion labels
├── val.csv           # Validation data
└── test.csv          # Test data
```

Format:
```csv
text,emotion
"हांव खूश आसा",happy
"हांव दुखी आसा",sad
"हांव रागीत आसा",angry
```

## 🎯 Step-by-Step: Translation Training

### 1. Check Your Translation Data
```bash
# Do you have translation pairs?
ls data/translation/

# If not, generate them
python scripts/generate_translation_pairs.py
```

### 2. Upload to Kaggle
```bash
# Package the data
cd data/translation
zip -r ../../konkani_translation_data.zip .
cd ../..

# Upload via web UI or CLI
kaggle datasets create -p data/translation -r zip
```

### 3. Run Training Notebook
- Open `notebooks/KAGGLE_TRANSLATION_TRAINING.ipynb`
- Update dataset path
- Run all cells
- Wait ~2-3 hours

### 4. Download Trained Model
```bash
kaggle kernels output YOUR_USERNAME/translation-training -p ./translation_outputs
```

## 🎯 Step-by-Step: Emotion Training

### 1. Generate Emotion Data
```bash
# Run the generation notebook
jupyter notebook notebooks/GENERATE_EMOTION_DATA.ipynb

# Or use the script
python scripts/generate_emotion_data.py
```

### 2. Upload to Kaggle
```bash
# Package the data
cd data/emotion
zip -r ../../konkani_emotion_data.zip .
cd ../..

# Upload
kaggle datasets create -p data/emotion -r zip
```

### 3. Run Training Notebook
- Open `notebooks/KAGGLE_EMOTION_TRAINING.ipynb`
- Update dataset path
- Run all cells
- Wait ~1-2 hours

### 4. Download Trained Model
```bash
kaggle kernels output YOUR_USERNAME/emotion-training -p ./emotion_outputs
```

## 📊 Monitoring All Training Sessions

### Check Status
```bash
# List all your running notebooks
kaggle kernels list --mine

# Check specific notebook
kaggle kernels status YOUR_USERNAME/translation-training
kaggle kernels status YOUR_USERNAME/emotion-training
```

### View Logs
```bash
# Get latest output
kaggle kernels output YOUR_USERNAME/translation-training
kaggle kernels output YOUR_USERNAME/emotion-training
```

## 💡 Pro Tips

### 1. Use Different Accounts for Parallel Training
- Account A: ASR (main account)
- Account B: Translation (secondary email)
- Account C: Emotion (tertiary email)

### 2. Optimize Training Order
If using one account:
1. **Start with Emotion** (fastest, 1-2h)
2. **Then Translation** (medium, 2-3h)
3. **Finally ASR** (longest, 6-8h)

This way you get quick wins first!

### 3. Download Models Immediately
After each training completes:
```bash
# Download right away before session expires
kaggle kernels output USERNAME/NOTEBOOK -p ./outputs
```

### 4. Monitor GPU Quota
```bash
# Check remaining hours
# Go to: https://www.kaggle.com/settings
# Look for "GPU Quota"
```

## 🔧 Troubleshooting

### "Dataset not found"
- Make sure you added the dataset to your notebook
- Check the dataset path in the notebook
- Verify dataset is public or shared with you

### "Out of memory"
- Reduce batch size in the notebook
- Use gradient accumulation
- Enable mixed precision training

### "Session expired"
- Kaggle has 12-hour limit per session
- Download checkpoints before expiry
- Resume from last checkpoint if needed

## 📈 Expected Results

### Translation Model
- **BLEU Score**: 25-35 (good for low-resource)
- **Training Time**: 2-3 hours
- **Model Size**: ~300MB

### Emotion Model
- **Accuracy**: 70-85%
- **Training Time**: 1-2 hours
- **Model Size**: ~500MB

## 🎉 After All Training Completes

You'll have three trained models:

```
models/
├── asr/
│   └── best_model.pt              # ASR model ✅
├── translation/
│   └── konkani_english_model/     # Translation model
└── emotion/
    └── emotion_classifier/        # Emotion model
```

## 🚀 Next Steps

1. **Test all models** on real data
2. **Evaluate performance** (WER, BLEU, Accuracy)
3. **Deploy** for inference
4. **Create demo** combining all three

## 📝 Quick Reference

### Upload Dataset
```bash
kaggle datasets create -p /path/to/data -r zip
```

### Create Notebook
```bash
kaggle kernels push -p /path/to/notebook
```

### Download Outputs
```bash
kaggle kernels output USERNAME/NOTEBOOK -p ./outputs
```

### Check Status
```bash
kaggle kernels list --mine
```

---

**Ready to train?** Start with the easiest (emotion) to get familiar with the process!
