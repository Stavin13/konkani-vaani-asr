# Kaggle Retraining Guide - Complete Setup

## Why Kaggle?

✅ **Free GPU** (P100/T4) - 30 hours/week  
✅ **No local setup** - Everything in browser  
✅ **Auto-save** - Checkpoints saved automatically  
✅ **Fast training** - 8-12 hours vs 50+ hours on CPU  

---

## Step 1: Prepare Data Locally (30 minutes)

### 1.1 Create Data Package

```bash
cd /Volumes/data&proj/konkani

# Create a zip with all necessary data
zip -r konkani_complete_data.zip \
  KonkaniRawSpeechCorpus/ \
  data/konkani-asr-v0/ \
  data/vocab.json \
  models/ \
  data/audio_processing/ \
  -x "*.pyc" "*__pycache__*" "*.git*"
```

**Expected size:** ~5-10 GB (depending on audio format)

### 1.2 Alternative: Smaller Package (if too large)

If the zip is too large (>20GB), create two separate packages:

```bash
# Package 1: Raw corpus only
zip -r konkani_raw_corpus.zip KonkaniRawSpeechCorpus/

# Package 2: Code and existing data
zip -r konkani_code_data.zip \
  data/ \
  models/ \
  -x "*.pyc" "*__pycache__*"
```

---

## Step 2: Upload to Kaggle (15 minutes)

### 2.1 Create Kaggle Dataset

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload `konkani_complete_data.zip`
4. Title: **"Konkani ASR Complete Data"**
5. Make it **Private** (your data)
6. Click **"Create"**

**Upload time:** 10-30 minutes depending on size and internet speed

### 2.2 Verify Upload

Once uploaded, check:
- Dataset shows correct size
- Files are accessible
- No corruption errors

---

## Step 3: Create Kaggle Notebook (5 minutes)

### 3.1 Create New Notebook

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Title: **"KonkaniVani ASR - Retraining with Testing"**
4. Settings:
   - **Accelerator:** GPU P100 or T4
   - **Internet:** ON (for pip installs)
   - **Persistence:** Files only

### 3.2 Add Your Dataset

1. In notebook, click **"+ Add Data"** (right sidebar)
2. Search for your dataset: "Konkani ASR Complete Data"
3. Click **"Add"**
4. Dataset will be available at `/kaggle/input/konkani-asr-complete-data/`

---

## Step 4: Upload Notebook Code (5 minutes)

### Option A: Copy-Paste (Easiest)

1. Open `notebooks/KAGGLE_RETRAIN_WITH_TESTING.ipynb` locally
2. Copy each cell
3. Paste into Kaggle notebook cells

### Option B: Upload Notebook

1. In Kaggle, click **"File" → "Upload Notebook"**
2. Select `notebooks/KAGGLE_RETRAIN_WITH_TESTING.ipynb`
3. Notebook will be imported with all cells

---

## Step 5: Configure Paths (2 minutes)

In the notebook, update these paths to match your dataset:

```python
# Cell: Check if data is available
DATA_ROOT = Path('/kaggle/input/konkani-asr-complete-data')  # ← Your dataset name

# Cell: Prepare data
corpus_dir='/kaggle/input/konkani-asr-complete-data/KonkaniRawSpeechCorpus/Data'
existing_dir='/kaggle/input/konkani-asr-complete-data/data/konkani-asr-v0/splits/manifests'
```

---

## Step 6: Start Training (8-12 hours)

### 6.1 Run All Cells

1. Click **"Run All"** or run cells sequentially
2. Monitor output for:
   - Data preparation progress
   - Training loss decreasing
   - Test results every 5 epochs

### 6.2 Expected Output

```
================================================================================
TRANSCRIPTION TEST - EPOCH 5
================================================================================

Metrics:
  Blank prob: 96.2%
  Unique tokens: 8.4
  Status: ❌ Not yet

================================================================================
TRANSCRIPTION TEST - EPOCH 10
================================================================================

Metrics:
  Blank prob: 88.5%
  Unique tokens: 18.2
  Status: ❌ Not yet

================================================================================
TRANSCRIPTION TEST - EPOCH 20
================================================================================

Metrics:
  Blank prob: 72.3%
  Unique tokens: 42.1
  Status: ✅ WORKING!

Samples:
  [1] GT: राम आनी लक्ष्मण दीनबंधू जावन हांसत खेळत वनवासाक गेले
      PR: राम आनी लक्ष्मण दीन हांसत खेळत वनवासाक गेले
      Blank: 68.2%, Tokens: 45
```

### 6.3 Monitor Progress

- **Epoch 1-10:** High blank prob (normal)
- **Epoch 10-20:** Blank prob dropping (good sign!)
- **Epoch 20-30:** Should see "✅ WORKING!"
- **Epoch 30-50:** Improving transcriptions
- **Epoch 50-100:** Refinement

---

## Step 7: Download Checkpoints

### 7.1 During Training

Kaggle auto-saves to `/kaggle/working/checkpoints/`

### 7.2 After Training

1. In notebook, run download cell:
   ```python
   from IPython.display import FileLink
   FileLink('/kaggle/working/checkpoints/best_model.pt')
   ```

2. Or use Kaggle Output:
   - Checkpoints saved to **"Output"** tab
   - Click **"Download"** to get all files

### 7.3 Download via Kaggle API (Alternative)

```bash
# On your local machine
kaggle kernels output <your-username>/konkanivani-asr-retraining -p ./kaggle_outputs
```

---

## Step 8: Test Locally

Once downloaded:

```bash
# Copy checkpoint to your project
cp kaggle_outputs/best_model.pt checkpoints/

# Test transcription
python scripts/test_best_model.py \
  --checkpoint checkpoints/best_model.pt \
  --max_files 10
```

---

## Troubleshooting

### Issue: "Out of Memory"

**Solution:**
```python
# Reduce batch size
CONFIG['batch_size'] = 8  # or 4

# Enable gradient accumulation
CONFIG['gradient_accumulation_steps'] = 4
```

### Issue: "Dataset not found"

**Solution:**
1. Check dataset is added to notebook (right sidebar)
2. Verify path: `/kaggle/input/<your-dataset-name>/`
3. List contents: `!ls /kaggle/input/`

### Issue: "Kernel disconnected"

**Solution:**
- Kaggle has 9-hour session limit
- Save checkpoints frequently
- Resume from last checkpoint:
  ```python
  CONFIG['resume_from_checkpoint'] = '/kaggle/working/checkpoints/checkpoint_epoch_40.pt'
  ```

### Issue: "GPU quota exceeded"

**Solution:**
- Kaggle gives 30 GPU hours/week
- Check usage: https://www.kaggle.com/settings
- Wait for quota reset (weekly)
- Or use CPU (slower but works)

---

## Time Estimates

| Task | Time |
|------|------|
| Prepare data locally | 30 min |
| Upload to Kaggle | 10-30 min |
| Setup notebook | 10 min |
| **Training (GPU)** | **8-12 hours** |
| Download checkpoints | 5 min |
| **Total** | **~10-14 hours** |

---

## Cost

**FREE!** Kaggle provides:
- 30 GPU hours/week (free)
- 20 GB storage (free)
- Unlimited CPU hours (free)

---

## Tips for Success

### 1. Monitor Regularly
Check notebook every 2-3 hours to ensure:
- Training is progressing
- No errors occurred
- GPU is being utilized

### 2. Save Checkpoints
```python
# Save every 5 epochs
CONFIG['save_every_n_epochs'] = 5
```

### 3. Test Early
If by epoch 30 blank prob is still >90%:
- Stop training
- Check configuration
- Verify data is loading correctly

### 4. Use Kaggle Discussions
If stuck, ask in:
- Kaggle notebook comments
- Kaggle forums
- Your dataset discussion

---

## Expected Results

### After 50 Epochs
- **Blank prob:** 40-60%
- **CER:** 30-50%
- **Status:** ✅ Working model

### After 100 Epochs
- **Blank prob:** 30-40%
- **CER:** 20-35%
- **Status:** ✅ Production ready

---

## Next Steps After Training

1. **Evaluate on test set**
   ```bash
   python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pt
   ```

2. **Test on real audio**
   ```bash
   python scripts/test_best_model.py --checkpoint checkpoints/best_model.pt --audio your_audio.wav
   ```

3. **Deploy**
   - Create inference API
   - Optimize for production
   - Share with users

---

## Quick Start Checklist

- [ ] Zip data locally
- [ ] Upload to Kaggle Datasets
- [ ] Create Kaggle Notebook
- [ ] Add dataset to notebook
- [ ] Enable GPU accelerator
- [ ] Copy notebook code
- [ ] Update paths
- [ ] Run all cells
- [ ] Monitor progress
- [ ] Download checkpoints
- [ ] Test locally

---

## Support

If you encounter issues:
1. Check Kaggle notebook output for errors
2. Review this guide's troubleshooting section
3. Check Kaggle documentation: https://www.kaggle.com/docs
4. Ask in Kaggle forums

Good luck with training! 🚀
