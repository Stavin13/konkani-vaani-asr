# Dual GPU Training - Complete Setup

## 🚀 Maximum Speed Configuration

### What You Have Now:

**File:** `KAGGLE_DUAL_GPU_PIPELINE.ipynb`

**Magic:** Trains emotion AND translation models **simultaneously** on 2 GPUs!

---

## How It Works

### GPU Assignment:
- **GPU 0 (cuda:0):** Emotion model training
- **GPU 1 (cuda:1):** Translation model training

### Parallel Execution:
- Uses Python `threading` module
- Both models train at the same time
- No waiting for one to finish!

### Timeline:
```
Time 0:00 → Load data & generate labels (30 min)
         ↓
Time 0:30 → START PARALLEL TRAINING
         ├─ GPU 0: Emotion (15 epochs, 1.5 hours) ──┐
         └─ GPU 1: Translation (10 epochs, 1.5 hours) ─┤
                                                        ↓
Time 2:00 → BOTH DONE! Save models
```

**Total: ~2 hours** (vs 3-4 hours sequential)

---

## Complete Training Strategy

### Account 1: ASR (Main)
- **Notebook:** `KAGGLE_TRAINING.ipynb`
- **Status:** ✅ Already running!
- **GPU:** Dual T4 x2
- **Time:** 8.5 hours
- **Output:** Konkani ASR model

### Account 2: Emotion + Translation (Parallel)
- **Notebook:** `KAGGLE_DUAL_GPU_PIPELINE.ipynb`
- **Status:** 🟢 Ready to start!
- **GPU:** Dual T4 x2
- **Time:** 2 hours
- **Output:** Emotion + Translation models

### Result:
- **Total time:** 8.5 hours (everything finishes together!)
- **Total models:** 3 custom models
- **Total accounts:** 2 (not 3!)
- **GPU efficiency:** 100% (all 4 GPUs working)

---

## Setup Instructions

### 1. Prepare Data
Upload your Konkani text to Kaggle:
- From ASR transcripts (`train.json`)
- Or text corpus (`konkani_corpus.txt`)

### 2. Create Notebook
```
1. Go to kaggle.com/code
2. New Notebook
3. Copy KAGGLE_DUAL_GPU_PIPELINE.ipynb content
4. Add your text dataset
5. Update DATASET_PATH in Step 3
```

### 3. Enable Dual GPU
```
Settings → Accelerator → GPU T4 x2
```
**Important:** Must select "T4 x2" not just "T4"!

### 4. Run
```
1. Run all cells
2. Turn off internet after Step 2
3. Watch the magic happen!
```

---

## What You'll See

### Console Output:
```
==================================================
DUAL GPU SETUP
==================================================
Available GPUs: 2

GPU 0: Tesla T4
Memory: 15.00 GB

GPU 1: Tesla T4
Memory: 15.00 GB

✅ Dual GPU detected!
GPU 0: Emotion model training
GPU 1: Translation model training
==================================================

...

==================================================
🚀 STARTING PARALLEL TRAINING ON DUAL GPUs
==================================================

[GPU 0] Starting emotion training...
[GPU 1] Starting translation training...

[GPU 0] Emotion Epoch 1/15: Train Loss: 1.2345, Val Acc: 65.23%
[GPU 1] Translation Epoch 1/10: Train Loss: 3.4567, Val Loss: 3.2109

[GPU 0] Emotion Epoch 2/15: Train Loss: 0.9876, Val Acc: 72.45%
[GPU 1] Translation Epoch 2/10: Train Loss: 2.8765, Val Loss: 2.7654

...

[GPU 0] ✅ Emotion training complete!
[GPU 1] ✅ Translation training complete!

==================================================
✅ PARALLEL TRAINING COMPLETE!
Total time: 1.8 hours
==================================================
```

---

## Technical Details

### Emotion Model (GPU 0):
- **Architecture:** Bidirectional LSTM with attention
- **Vocab size:** ~10,000 Konkani words
- **Embedding:** 128 dim
- **Hidden:** 256 dim
- **Layers:** 2
- **Parameters:** ~2-3M
- **Training:** 15 epochs, ~1.5 hours
- **Batch size:** 64

### Translation Model (GPU 1):
- **Architecture:** Seq2Seq with attention
- **Encoder:** Bidirectional LSTM (256 hidden, 2 layers)
- **Decoder:** LSTM with attention (512 hidden, 2 layers)
- **Vocab:** ~10,000 Konkani + ~10,000 English
- **Embedding:** 256 dim
- **Parameters:** ~15-20M
- **Training:** 10 epochs, ~1.5 hours
- **Batch size:** 32

### Why Different Batch Sizes?
- Emotion model is smaller → can handle larger batches (64)
- Translation model is larger → needs smaller batches (32)
- Both optimized for T4 memory (15GB)

---

## Advantages

### vs Sequential Training:
- ⏱️ **50% faster** (2 hours vs 3-4 hours)
- 🎯 **Same quality** (same epochs, same data)
- 💪 **Better GPU usage** (both GPUs working)

### vs Separate Notebooks:
- 📝 **Easier management** (one notebook, not two)
- 🔄 **Shared data** (load once, use twice)
- 📦 **One download** (both models together)

### vs Fine-tuning:
- 🎨 **Truly custom** (built from scratch)
- 🔧 **Full control** (modify architecture easily)
- 📊 **Smaller models** (faster inference)

---

## Troubleshooting

### "Only 1 GPU detected"
- Check Kaggle settings: Must select "T4 x2" not "T4"
- Restart notebook and run Step 1 again

### "CUDA out of memory"
- Reduce batch sizes in Step 5
- Emotion: 64 → 32
- Translation: 32 → 16

### "Threading not working"
- Models will train sequentially (still works!)
- Just takes longer (~3 hours instead of 2)

---

## After Training

### Download Models:
1. Go to Output tab
2. Download `/kaggle/working/konkani_emotion_model/`
3. Download `/kaggle/working/konkani_translation_model/`

### Test Locally:
```python
# Load emotion model
emotion_model = EmotionLSTM(...)
emotion_model.load_state_dict(torch.load('model.pt'))

# Load translation model
translation_model = Seq2Seq(...)
translation_model.load_state_dict(torch.load('model.pt'))
```

### Full Pipeline:
```
Audio → ASR → Konkani text → Emotion detector
                           → Translator → English
```

---

## Summary

✅ **One notebook** trains both models
✅ **Dual GPU** for parallel training
✅ **2 hours** total time
✅ **Custom models** built from scratch
✅ **Ready to deploy** with ASR model

**Next step:** Run `KAGGLE_DUAL_GPU_PIPELINE.ipynb` on Account 2!
