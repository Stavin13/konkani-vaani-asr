# Complete Retraining Plan with Periodic Testing

## Summary

Your current model predicts **98% blank tokens** after 50 epochs. This is due to:
- CTC weight too low (0.3 instead of 0.8)
- Insufficient data (21h vs 84h available)
- No monitoring of actual transcription quality

## Solution: Retrain with New Data + Fixed Config + Periodic Testing

---

## Step 1: Prepare Data (1-2 hours)

### Current Data
- 2,549 samples × 30s = **21.2 hours**

### New Data Available
- KonkaniRawSpeechCorpus: 72,937 samples × 4.2s = **84.1 hours**
- High quality, labeled, diverse speakers

### Prepare Combined Dataset

```bash
# Create manifests from raw corpus
python scripts/prepare_raw_corpus_data.py
```

This will:
1. Parse all 72,937 audio files and transcriptions
2. Create train/val/test splits (80/10/10)
3. Combine with existing data
4. Output: `data/konkani-combined/manifests/`

**Expected Result:**
- Train: ~60,000 samples (~70h)
- Val: ~7,500 samples (~9h)
- Test: ~7,500 samples (~9h)
- **Total: ~88 hours of labeled data**

---

## Step 2: Configure Training with Testing

### Key Configuration Changes

**Fixed Issues:**
```yaml
# OLD (broken)
ctc_weight: 0.3
learning_rate: 0.0001
grad_clip: null
testing: disabled

# NEW (fixed)
ctc_weight: 0.8          # Focus on CTC
learning_rate: 0.0003    # Higher LR
grad_clip: 5.0           # Prevent exploding gradients
testing: every 5 epochs  # Monitor progress
```

### Use Enhanced Config

```bash
# Training will use this config
config/training_config_with_testing.yaml
```

---

## Step 3: Train with Periodic Testing

### Start Training

```bash
python training_scripts/train_konkanivani_asr.py \
  --config config/training_config_with_testing.yaml \
  --epochs 100
```

### What Happens During Training

**Every 5 epochs**, the script will:
1. Test on 5 sample audio files
2. Measure:
   - Blank token probability (should decrease)
   - Unique tokens predicted (should increase)
   - Actual transcriptions (visual inspection)
3. Save results to `checkpoints/test_results_epoch_N.json`
4. Print progress report

**Example Output:**
```
================================================================================
TRANSCRIPTION TEST - EPOCH 20
================================================================================

Overall Metrics:
  Avg blank probability: 65.3%
  Avg unique tokens: 28.4
  Status: ✅ WORKING!

Sample Transcriptions:
  [1] segment_001548.wav
      Ground truth: राम आनी लक्ष्मण दीनबंधू जावन हांसत खेळत वनवासाक गेले
      Prediction:   राम आनी लक्ष्मण दीन हांसत खेळत वनवासाक गेले
      Blank prob: 62.1%, Tokens: 32
```

---

## Step 4: Expected Progress Timeline

### Epoch 1-10: Initial Learning
- **Blank prob:** 95-98%
- **Unique tokens:** 3-10
- **Status:** Not working yet (normal)
- **What's happening:** Model learning basic patterns

### Epoch 10-20: Starting to Work
- **Blank prob:** 80-90%
- **Unique tokens:** 10-30
- **Status:** Some characters appearing
- **What's happening:** CTC learning character alignments

### Epoch 20-40: Recognizable Words
- **Blank prob:** 50-80%
- **Unique tokens:** 30-80
- **Status:** ✅ Working! Words recognizable
- **What's happening:** Model learning word patterns

### Epoch 40-100: Refinement
- **Blank prob:** 30-50%
- **Unique tokens:** 80-150
- **Status:** ✅ Good transcriptions
- **What's happening:** Improving accuracy, reducing errors

---

## Step 5: Monitor and Adjust

### Check Progress

```bash
# Test any checkpoint manually
python scripts/train_with_periodic_testing.py \
  --checkpoint checkpoints/checkpoint_epoch_20.pt \
  --num_samples 10
```

### View Training Logs

```bash
# TensorBoard
tensorboard --logdir logs

# Check test results
cat checkpoints/test_results_epoch_20.json
```

### Early Stopping

If model is working well by epoch 30-40:
- Consider stopping early
- Reduce learning rate and continue
- Or train to epoch 100 for best results

---

## Step 6: Compare Results

### Before (Current Model)
```
Epoch: 27
Val Loss: 3.34
Blank prob: 98.57%
Unique tokens: 3
Status: ❌ Not working
```

### After (Expected with New Training)
```
Epoch: 40
Val Loss: 1.2-2.0
Blank prob: 40-60%
Unique tokens: 80-120
Status: ✅ Working!
CER: 20-40%
```

---

## Estimated Time

### Data Preparation
- Parse corpus: 30 min
- Create manifests: 15 min
- **Total: ~1 hour**

### Training Time (depends on hardware)

**CPU only:**
- Per epoch: ~30-45 min
- 100 epochs: ~50-75 hours
- **Recommendation:** Use GPU or Kaggle

**GPU (T4/P100):**
- Per epoch: ~5-10 min
- 100 epochs: ~8-16 hours
- **Recommendation:** Kaggle free GPU

**GPU (V100/A100):**
- Per epoch: ~2-5 min
- 100 epochs: ~3-8 hours

### Testing Overhead
- Per test (every 5 epochs): ~1-2 min
- Total testing time: ~20-40 min
- **Negligible impact on training time**

---

## Quick Start Commands

```bash
# 1. Prepare data
python scripts/prepare_raw_corpus_data.py

# 2. Start training with testing
python training_scripts/train_konkanivani_asr.py \
  --config config/training_config_with_testing.yaml \
  --epochs 100

# 3. Monitor progress (in another terminal)
tensorboard --logdir logs

# 4. Test checkpoint manually
python scripts/train_with_periodic_testing.py \
  --checkpoint checkpoints/checkpoint_epoch_20.pt
```

---

## Success Criteria

### Minimum (Working Model)
- ✅ Blank prob < 80% by epoch 30
- ✅ Unique tokens > 30 by epoch 30
- ✅ Some recognizable words in transcriptions

### Good (Production Ready)
- ✅ Blank prob < 50% by epoch 50
- ✅ Unique tokens > 80 by epoch 50
- ✅ CER < 40%
- ✅ Most words recognizable

### Excellent (State of Art)
- ✅ Blank prob < 30% by epoch 100
- ✅ Unique tokens > 120 by epoch 100
- ✅ CER < 25%
- ✅ High quality transcriptions

---

## Troubleshooting

### If blank prob stays > 95% after epoch 20:
1. Check CTC weight is 0.8 (not 0.3)
2. Verify blank_id in vocab is correct
3. Increase learning rate to 5e-4
4. Check data is loading correctly

### If loss not decreasing:
1. Reduce learning rate to 1e-4
2. Increase gradient clipping to 10.0
3. Check for NaN in gradients
4. Verify data preprocessing

### If out of memory:
1. Reduce batch_size to 8 or 4
2. Enable gradient_accumulation_steps: 4
3. Reduce max_duration to 10.0
4. Use mixed_precision: true

---

## Next Steps After Training

1. **Evaluate on test set**
   ```bash
   python scripts/evaluate_model.py \
     --checkpoint checkpoints/best_model.pt
   ```

2. **Test on real audio**
   ```bash
   python scripts/test_best_model.py \
     --checkpoint checkpoints/best_model.pt \
     --audio your_audio.wav
   ```

3. **Deploy for inference**
   - Use best checkpoint
   - Optimize for production
   - Create API endpoint

---

## Conclusion

**Yes, retraining with new data will make a HUGE difference!**

- 4x more data (84h vs 21h)
- Fixed training config (CTC weight 0.8)
- Periodic testing to monitor progress
- Expected to work by epoch 20-30

**Estimated total time:** 10-20 hours (mostly training)
**Expected result:** Working ASR model with 20-40% CER
