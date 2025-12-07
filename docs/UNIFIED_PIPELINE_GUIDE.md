# Dual GPU Emotion + Translation Pipeline

## One Notebook, Two Models, Two GPUs! 🚀🚀

**File:** `KAGGLE_DUAL_GPU_PIPELINE.ipynb`

**What it does:**
1. Loads your Konkani text data
2. Generates emotion labels using pre-trained model
3. Generates English translations using pre-trained model
4. **Trains BOTH models simultaneously on 2 GPUs:**
   - GPU 0: Custom LSTM emotion classifier
   - GPU 1: Custom Seq2Seq translation model
5. Saves both models

**Time:** ~2 hours on dual T4 x2 (vs 3-4 hours sequential!)

**Key Feature:** Parallel training using Python threading - both models train at the same time!

---

## Setup

### 1. Prepare Your Data
You need Konkani text (from ASR transcripts, corpus, etc.)

**Format options:**
- JSON manifest (like `train.json` from ASR)
- Plain text file (`konkani_corpus.txt`)

### 2. Upload to Kaggle
1. Create dataset: "Konkani Text Data"
2. Upload your text file(s)
3. Make it public or add to your account

### 3. Create Notebook
1. New Kaggle notebook
2. Copy content from `KAGGLE_DUAL_GPU_PIPELINE.ipynb`
3. Add your text dataset
4. Update `DATASET_PATH` in Step 3
5. **Enable GPU: T4 x2 (dual GPU required for parallel training)**

### 4. Run!
- Run all cells
- Turn off internet after dependencies install
- **Watch both GPUs train in parallel!**
- Wait ~2 hours (both models finish together)
- Download both models from Output tab

---

## What You Get

### Emotion Model
- **Location:** `/kaggle/working/konkani_emotion_model/`
- **Files:**
  - `model.pt` - Trained LSTM weights
  - `vocab.json` - Konkani vocabulary
  - `labels.json` - Emotion label mappings
- **Architecture:** Bidirectional LSTM with attention
- **Input:** Konkani text
- **Output:** Emotion label (joy, sadness, anger, etc.)

### Translation Model
- **Location:** `/kaggle/working/konkani_translation_model/`
- **Files:**
  - `model.pt` - Trained Seq2Seq weights
  - `src_vocab.json` - Konkani vocabulary
  - `tgt_vocab.json` - English vocabulary
- **Architecture:** Seq2Seq with attention
- **Input:** Konkani text
- **Output:** English translation

---

## Model Details

### Emotion Model (Custom LSTM)
```
Architecture:
- Embedding layer (128 dim)
- Bidirectional LSTM (256 hidden, 2 layers)
- Attention mechanism
- Dropout (0.3)
- Output layer (num_emotions)

Parameters: ~2-3M
Training: 15 epochs
```

### Translation Model (Custom Seq2Seq)
```
Encoder:
- Embedding (256 dim)
- Bidirectional LSTM (256 hidden, 2 layers)

Decoder:
- Embedding (256 dim)
- LSTM with attention (512 hidden, 2 layers)
- Output layer (vocab_size)

Parameters: ~15-20M
Training: 10 epochs
```

---

## Advantages of Dual GPU Pipeline

✅ **Single notebook** - easier to manage
✅ **Shared data loading** - load Konkani text once
✅ **Parallel training** - both models train simultaneously!
✅ **Dual GPU usage** - GPU 0 + GPU 1 working together
✅ **Faster** - ~2 hours total (vs 3-4 hours sequential)
✅ **One download** - both models in Output tab
✅ **Truly custom** - no pre-trained model fine-tuning
✅ **Python threading** - automatic parallel execution

---

## Parallel Training Strategy

### Option 1: Dual GPU Pipeline (BEST!)
- **Account 1:** ASR training (8.5 hours, dual T4 x2)
- **Account 2:** Emotion + Translation dual GPU pipeline (2 hours, dual T4 x2)
- **Total time:** 8.5 hours (everything done!)
- **GPUs used:** 4 total (2 per account)

### Option 2: Sequential Pipeline
- **Account 1:** ASR training (8.5 hours)
- **Account 2:** Emotion + Translation sequential (3-4 hours)
- **Total time:** 8.5 hours
- **GPUs used:** 2 total

### Option 3: Separate Notebooks
- **Account 1:** ASR training (8.5 hours)
- **Account 2:** Emotion training (1 hour)
- **Account 3:** Translation training (2-3 hours)
- **Total time:** 8.5 hours (but need 3 accounts)

**Dual GPU pipeline wins!** Fastest, fewer accounts, maximum GPU utilization!

---

## Quick Start

```bash
# 1. Upload your Konkani text to Kaggle as dataset

# 2. Create new Kaggle notebook

# 3. Copy KAGGLE_DUAL_GPU_PIPELINE.ipynb content

# 4. Update DATASET_PATH in Step 3

# 5. Enable GPU: T4 x2 (DUAL GPU!)

# 6. Run all cells

# 7. Watch both GPUs train in parallel!

# 8. Wait ~2 hours

# 9. Download models from Output tab
```

---

## Timeline

| Step | Time | Description |
|------|------|-------------|
| Setup | 2 min | Install dependencies, check dual GPU |
| Load data | 1 min | Load Konkani texts |
| Generate emotion labels | 10-15 min | Use pre-trained model |
| Generate translations | 20-30 min | Use pre-trained model |
| **Parallel Training** | **1.5-2 hours** | **Both models train simultaneously!** |
| - GPU 0: Emotion model | (1.5 hours) | Custom LSTM (15 epochs) |
| - GPU 1: Translation model | (1.5 hours) | Custom Seq2Seq (10 epochs) |
| Save models | 1 min | Save both models |
| **TOTAL** | **~2 hours** | Both models ready! |

---

## Notes

- **Custom models:** Built from scratch, not fine-tuned
- **No pre-trained weights:** Pure Konkani-specific learning
- **Smaller models:** Faster inference, easier deployment
- **Full control:** Modify architecture as needed

---

## Next Steps After Training

1. Download both models from Kaggle Output tab
2. Test locally with your Konkani text
3. Integrate into your ASR pipeline:
   - ASR → Konkani text
   - Emotion model → Detect emotion
   - Translation model → Translate to English
4. Deploy all 3 models together!

---

## Questions?

Check the notebook comments - each step is documented!
