# Kaggle ASR Retraining - Quick Fix Guide

## Problem
Your ASR model outputs **98% blank tokens** and can't transcribe audio.

## Root Cause
- ❌ CTC weight too low (0.3 instead of 0.8)
- ❌ Insufficient data (21h instead of 88h available)
- ❌ No monitoring during training

## Solution: 3 Critical Fixes

### Fix #1: CTC Weight (MOST IMPORTANT)
```python
# OLD (broken)
ctc_weight: 0.3

# NEW (fixed)
ctc_weight: 0.8  # 🔥 This makes transcription work!
```

**Why**: CTC loss teaches the model character alignment. At 0.3, the model ignores it and just outputs blanks.

### Fix #2: Use Full Dataset
```bash
# Current: 21 hours (2,549 samples)
# Available: 84 hours (72,937 samples in KonkaniRawSpeechCorpus)
# Total: 88 hours combined

# Prepare full dataset locally:
python scripts/prepare_raw_corpus_data.py
```

### Fix #3: Add Testing
Monitor blank probability every 5 epochs to see when model starts working.

---

## Quick Start (3 Steps)

### Step 1: Prepare Data Locally (30 min)

```bash
cd /Volumes/data&proj/konkani

# Prepare full dataset
python scripts/prepare_raw_corpus_data.py

# Package for Kaggle
bash scripts/package_for_kaggle.sh
```

This creates: `kaggle_package/konkani_complete_data.zip` (~5-10 GB)

### Step 2: Upload to Kaggle (15 min)

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload `konkani_complete_data.zip`
4. Title: "Konkani ASR Complete Dataset"
5. Make it **Private**

### Step 3: Run Fixed Notebook (8-12 hours)

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload: `notebooks/KAGGLE_RETRAIN_FIXED.ipynb`
4. Settings:
   - Accelerator: **GPU P100 or T4**
   - Internet: **ON**
5. Add your dataset (right sidebar)
6. Click "Run All"

---

## What to Expect

### During Training

| Epoch | Blank Prob | Status | What's Happening |
|-------|------------|--------|------------------|
| 1-10 | 95-98% | ❌ Not yet | Learning basics |
| 10-20 | 80-90% | 🟡 Starting | Characters appearing |
| 20-40 | 50-80% | ✅ **WORKING!** | Recognizable words |
| 40-100 | 30-50% | ✅ Good | Refinement |

### Test Output (Epoch 30)
```
================================================================================
TRANSCRIPTION TEST - EPOCH 30
================================================================================

Overall Metrics:
  Avg blank probability: 58.3%  ← Should be < 80%
  Avg unique tokens: 45.2       ← Should be > 30
  Status: ✅ WORKING!

Sample Transcriptions:
  [1] Ground truth: राम आनी लक्ष्मण दीनबंधू जावन हांसत खेळत वनवासाक गेले
      Prediction:   राम आनी लक्ष्मण दीन हांसत खेळत वनवासाक गेले
      Blank prob: 55.2%, Tokens: 48
```

---

## Configuration Comparison

| Setting | Old (Broken) | New (Fixed) | Impact |
|---------|--------------|-------------|--------|
| **CTC Weight** | 0.3 | **0.8** | 🔥 Critical - enables transcription |
| **Learning Rate** | 0.0001 | **0.0003** | Faster learning |
| **Gradient Clip** | None | **5.0** | Prevents explosion |
| **Data Size** | 21h | **88h** | 4x more training data |
| **Testing** | None | **Every 5 epochs** | Monitor progress |
| **Epochs** | 50 | **100** | More training time |

---

## Alternative: Quick Fix Without Re-upload

If you already have a Kaggle notebook running, just change the training command:

### Find this cell:
```python
!python training_scripts/train_konkanivani_asr.py \
    --train_manifest data/train.json \
    --val_manifest data/val.json \
    --learning_rate 0.0001 \
    --ctc_weight 0.3 \
    --num_epochs 50
```

### Change to:
```python
!python training_scripts/train_konkanivani_asr.py \
    --train_manifest data/train.json \
    --val_manifest data/val.json \
    --learning_rate 0.0003 \
    --ctc_weight 0.8 \
    --grad_clip 5.0 \
    --num_epochs 100
```

**Note**: This won't add the full dataset, but the CTC weight fix alone should help significantly.

---

## Troubleshooting

### "Out of Memory"
```python
# Reduce batch size
batch_size: 1
gradient_accumulation_steps: 8
```

### "Still 95% blanks after epoch 30"
Check:
1. CTC weight is actually 0.8 (print it in training loop)
2. Blank token ID is correct in vocab
3. Data is loading properly (check a few samples)

### "Training too slow"
- Use GPU T4 or P100 (not CPU)
- Enable mixed precision: `mixed_precision: true`
- Reduce validation frequency

---

## Success Criteria

### Minimum (Working Model)
- ✅ Blank prob < 80% by epoch 30
- ✅ Unique tokens > 30
- ✅ Some recognizable words

### Good (Production Ready)
- ✅ Blank prob < 50% by epoch 50
- ✅ CER < 40%
- ✅ Most words correct

### Excellent
- ✅ Blank prob < 30% by epoch 100
- ✅ CER < 25%
- ✅ High quality transcriptions

---

## Time Estimate

| Task | Time |
|------|------|
| Prepare data locally | 30 min |
| Upload to Kaggle | 15 min |
| Setup notebook | 5 min |
| **Training (GPU)** | **8-12 hours** |
| Download checkpoint | 5 min |
| **Total** | **~10-14 hours** |

---

## Files You Need

### Local Files
- ✅ `notebooks/KAGGLE_RETRAIN_FIXED.ipynb` (new notebook)
- ✅ `scripts/prepare_raw_corpus_data.py` (data prep)
- ✅ `scripts/package_for_kaggle.sh` (packaging)

### Upload to Kaggle
- ✅ `konkani_complete_data.zip` (full dataset)

### Download from Kaggle
- ✅ `best_model.pt` (trained model)

---

## Summary

**The #1 fix**: Change `ctc_weight` from 0.3 to 0.8

This single change will make your model start producing actual transcriptions instead of just blank tokens.

Combined with more data (88h) and proper monitoring, you should have a working model by epoch 20-30.

**Total effort**: ~1 hour prep + 10 hours training = **Working ASR model!**
