# Kaggle Retraining Instructions - Fixed Vocabulary

## 🎯 CRITICAL: This package contains the vocabulary fix!

**Problem**: Previous model used vocab_size=81, but data needs 193 characters
**Solution**: This model uses vocab_size=200, can predict all characters
**Expected**: 20-50% accuracy (vs previous 1%)

---

## Step 1: Upload Code & Model to Kaggle

### 1.1 Create Kaggle Dataset
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload these files:
   - `initial_model_vocab200.pt` (corrected model)
   - `vocab.json` (200 characters)
   - `models/konkanivani_asr.py`
   - `data/audio_processing/` (folder)

### 1.2 Dataset Settings
- **Title**: "KonkaniVani ASR - Fixed Vocabulary Model"
- **Subtitle**: "Model with vocab_size=200 (was 81)"
- **Description**: "Corrected ASR model that can predict all Konkani characters"
- **Visibility**: Private

---

## Step 2: Upload Training Data

### 2.1 Package Your Data
Run locally:
```bash
cd /Volumes/data&proj/konkani
./kaggle_retrain_fixed/package_data.sh
```

### 2.2 Upload Data to Kaggle
1. Create another dataset: "Konkani Training Data"
2. Upload `konkani_training_data.zip`
3. Make it private

---

## Step 3: Create Kaggle Notebook

### 3.1 Upload Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload `KonkaniVani_Fixed_Vocab_Training.ipynb`

### 3.2 Notebook Settings
- **Title**: "KonkaniVani ASR - Fixed Vocab Training"
- **Accelerator**: GPU P100 or T4
- **Internet**: ON

### 3.3 Add Datasets
In notebook sidebar:
1. Click "Add Data"
2. Add your "Fixed Vocabulary Model" dataset
3. Add your "Konkani Training Data" dataset

---

## Step 4: Update Notebook Paths

In the notebook, update these paths:

```python
# Update these paths to match your datasets:
model_path = '/kaggle/input/your-model-dataset/initial_model_vocab200.pt'
vocab_path = '/kaggle/input/your-model-dataset/vocab.json'
train_manifest = '/kaggle/input/your-data-dataset/train.json'
val_manifest = '/kaggle/input/your-data-dataset/val.json'
```

---

## Step 5: Run Training

### 5.1 Expected Timeline
- **Setup**: 5-10 minutes
- **Training**: 2-3 hours for 50 epochs
- **Total**: ~3-4 hours

### 5.2 Expected Results
- **Epoch 5**: See actual Devanagari characters (not "अध tस")
- **Epoch 10**: ~10-20% accuracy
- **Epoch 20**: ~20-35% accuracy  
- **Epoch 30**: ~30-45% accuracy
- **Epoch 50**: ~40-60% accuracy

### 5.3 Success Indicators
✅ **Good signs**:
- Predictions contain real Konkani words
- Accuracy > 10% by epoch 10
- Validation loss < 2.5

❌ **Bad signs** (if still happening):
- Still predicting "अध tस" patterns
- Accuracy < 5% after 20 epochs
- Contact for debugging

---

## Step 6: Download Results

After training:
1. Download checkpoints from `/kaggle/working/`
2. Test locally with: `python scripts/test_asr_latest.py`
3. Should see **dramatically better results**!

---

## 🚨 IMPORTANT NOTES

1. **Vocabulary Size**: Model MUST use vocab_size=200 (not 81)
2. **Expected Improvement**: 20-50x better accuracy
3. **Training Time**: Be patient, 50 epochs needed for good results
4. **GPU Hours**: Uses ~3-4 hours of your weekly quota

---

## Troubleshooting

### Issue: "Model vocab size mismatch"
**Solution**: Ensure you're using `initial_model_vocab200.pt`, not old checkpoints

### Issue: "Still getting 1% accuracy"
**Solution**: Check vocab.json has 200 characters, not 81

### Issue: "Out of memory"
**Solution**: Reduce batch_size from 8 to 4 or 2

---

## Contact

If you see the same poor results (1% accuracy), something went wrong.
The vocabulary fix should give **immediate and dramatic improvement**.

Good luck! 🚀
