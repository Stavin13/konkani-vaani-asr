# Kaggle Upload Guide - Optimized Training

## What Changed?

Your training at Epoch 15 showed overfitting:
- Train Loss: 5.32
- Val Loss: 9.59 (almost 2x!)

I've created an **optimized notebook** with better settings to fix this.

## Option 1: Quick Fix (Recommended)

### If Your Training is Still Running:

1. **Stop the Kaggle kernel** (click Stop button)
2. **Edit the training command** in your current notebook

Find this cell (Step 13):
```python
!python3 training_scripts/train_konkanivani_asr.py \
    --learning_rate 0.0005 \
```

Change to:
```python
!python3 training_scripts/train_konkanivani_asr.py \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --dropout 0.2 \
    --ctc_weight 0.5 \
```

3. **Restart from Step 13** (the training cell)

That's it! No need to re-upload anything.

## Option 2: Use New Optimized Notebook

### Upload the New Notebook:

1. **Download** from your local machine:
   ```
   notebooks/KAGGLE_TRAINING_OPTIMIZED.ipynb
   ```

2. **Go to Kaggle** → Your notebook → File → Upload Notebook

3. **Select** `KAGGLE_TRAINING_OPTIMIZED.ipynb`

4. **Run all cells**

## What to Upload as Dataset

You already have the dataset uploaded! Just use the same one:
```
/kaggle/input/konkani-asr-complete-dataset
```

### If You Need to Update the Dataset:

Only update if you want to include the new config file. Otherwise, the inline changes work fine.

**What's in your current dataset:**
```
kaggle_complete_dataset.zip containing:
├── training_scripts/
│   └── train_konkanivani_asr.py
├── models/
│   └── konkanivani_asr.py
├── data/
│   ├── audio_processing/
│   ├── konkani-asr-v0/
│   │   ├── audio/
│   │   └── splits/manifests/
│   └── vocab.json
├── archives/
│   └── checkpoint_epoch_15.pt
└── config/
    └── training_config_from_checkpoint15.yaml  ← Updated!
```

### To Update Dataset (Optional):

1. **On your local machine**, update the zip:
   ```bash
   cd /Users/stavinfernandes
   
   # Your zip already exists, just verify it has the updated config
   unzip -l kaggle_complete_dataset.zip | grep training_config
   ```

2. **If you want to include the new config**, recreate the zip:
   ```bash
   cd /Volumes/data&proj/konkani
   
   # Create updated package
   zip -r ~/kaggle_complete_dataset_v2.zip \
       training_scripts/ \
       models/ \
       data/audio_processing/ \
       data/vocab.json \
       data/konkani-asr-v0/splits/ \
       data/konkani-asr-v0/audio/ \
       archives/checkpoint_epoch_15.pt \
       config/training_config_from_checkpoint15.yaml
   ```

3. **Upload to Kaggle**:
   - Kaggle → Datasets → New Dataset
   - Upload `kaggle_complete_dataset_v2.zip`
   - Make it private
   - Update notebook to use new dataset path

## Comparison: Old vs New Settings

| Setting | Old (Overfitting) | New (Optimized) | Why? |
|---------|-------------------|-----------------|------|
| Learning Rate | 0.0005 | **0.0001** | Slower, more stable learning |
| Dropout | 0.1 | **0.2** | Stronger regularization |
| Weight Decay | 0.000001 | **0.0001** | 100x stronger penalty on large weights |
| CTC Weight | 0.3 | **0.5** | Better balance, CTC helps alignment |

## Expected Results

### After 5 Epochs (Epoch 20):
- Val loss should **stop increasing**
- Val/Train ratio should drop from 1.8 to ~1.5

### After 10 Epochs (Epoch 25):
- Val loss should **start decreasing**
- Model should generalize better
- CTC loss should improve

### Final (Epoch 50):
- Val/Train ratio should be < 1.3
- Better performance on real Konkani speech
- Lower Word Error Rate (WER)

## Quick Command Reference

### Check if training is running:
```python
!ps aux | grep train_konkanivani
```

### Stop training:
```python
import os
os.system("pkill -f train_konkanivani")
```

### Resume from best checkpoint:
```python
# Find checkpoint with lowest val loss
import torch
from pathlib import Path

checkpoints = sorted(Path('checkpoints').glob('checkpoint_epoch_*.pt'))
best_ckpt = None
best_val_loss = float('inf')

for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    val_loss = ckpt.get('val_loss', float('inf'))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_ckpt = ckpt_path

print(f"Best checkpoint: {best_ckpt}")
print(f"Val loss: {best_val_loss}")
```

## Troubleshooting

### "Training script doesn't accept --dropout argument"

Your training script might not have these parameters. Add them inline:

```python
# At the top of your training script
import argparse

parser = argparse.ArgumentParser()
# ... existing arguments ...
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.000001)
parser.add_argument('--ctc_weight', type=float, default=0.3)
```

### "Can't find checkpoint_epoch_15.pt"

Make sure it's in the right location:
```python
!find /kaggle/working -name "checkpoint_epoch_15.pt"
```

### "Out of memory"

Reduce batch size:
```python
BATCH_SIZE = 1
GRAD_ACCUM = 8
```

## Summary

**Easiest approach**: Just edit the training command in your current notebook to add the 4 new parameters. No need to re-upload anything!

**Best approach**: Use the new optimized notebook for cleaner code and better monitoring.

**Dataset**: Your current dataset is fine! Only update if you want the config file included (optional).
