# How to Update Your Running Kaggle Training

## Current Situation
- ✅ Training is running on Kaggle (Epoch 15/50)
- ❌ Using old config (causing overfitting)
- 📊 Val Loss (9.59) is 2x Train Loss (5.32)

## Quick Decision Guide

### Should I Stop Now?

**STOP if:**
- Val loss is increasing each epoch
- You're past epoch 20 and overfitting is getting worse
- You want to save GPU hours

**KEEP RUNNING if:**
- You're almost done (epoch 40+)
- Val loss is stable (not increasing)
- You have plenty of GPU hours left

## Option 1: Stop and Restart (Recommended)

### Step 1: Stop Kaggle Kernel
1. Go to your Kaggle notebook
2. Click "Stop" or "Interrupt" button
3. **Download checkpoint_epoch_15.pt** from the output

### Step 2: Update Files on Kaggle

#### Method A: Upload Updated Config (Easiest)
1. Download the updated config from your local machine:
   ```
   config/training_config_from_checkpoint15.yaml
   ```

2. In Kaggle, go to "Add Data" → "Upload"
3. Upload the new config file
4. Update the notebook to use the new config path

#### Method B: Edit Config in Kaggle Directly
Add this cell at the beginning of your notebook:

```python
# Update training config to reduce overfitting
import yaml

config = {
    'model': {
        'vocab_size': 200,
        'input_dim': 80,
        'd_model': 256,
        'encoder_layers': 12,
        'decoder_layers': 6,
        'num_heads': 4,
        'conv_kernel_size': 31,
        'dropout': 0.2  # Increased from 0.1
    },
    'training': {
        'learning_rate': 0.0001,  # Reduced from 0.0005
        'weight_decay': 0.0001,   # Increased from 0.000001
        'grad_clip': 5.0,
        'ctc_weight': 0.5,  # Increased from 0.3
        'batch_size': 2,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'num_epochs': 50,
        'save_every': 5
    },
    'data': {
        'train_manifest': 'data/konkani-asr-v0/splits/manifests/train.json',
        'val_manifest': 'data/konkani-asr-v0/splits/manifests/val.json',
        'vocab_file': 'data/vocab.json',
        'num_workers': 0
    },
    'paths': {
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'resume_from': 'checkpoint_epoch_15.pt'
    },
    'device': 'cuda'
}

# Save updated config
with open('config_optimized.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ Updated config saved!")
```

### Step 3: Modify Training Command

Change your training command from:
```python
!python3 training_scripts/train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file data/vocab.json \
    --checkpoint_dir checkpoints \
    --resume_from checkpoint_epoch_15.pt \
    --batch_size 2 \
    --learning_rate 0.0005 \
    --num_epochs 50
```

To:
```python
!python3 training_scripts/train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file data/vocab.json \
    --checkpoint_dir checkpoints \
    --resume_from checkpoint_epoch_15.pt \
    --batch_size 2 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --dropout 0.2 \
    --ctc_weight 0.5 \
    --num_epochs 50
```

### Step 4: Restart Training
1. Click "Run All" or run cells sequentially
2. Training will resume from epoch 15 with better settings

## Option 2: Let It Finish, Then Retrain

### What to Do:
1. Let current training complete
2. Note which epoch had the **lowest validation loss**
3. Download that checkpoint
4. Start a new Kaggle session with updated config
5. Resume from the best checkpoint

### How to Find Best Checkpoint:
Look at your training logs for the epoch with lowest val loss:
```
Epoch 12: Val Loss: 8.2  ← Best so far
Epoch 13: Val Loss: 8.5
Epoch 14: Val Loss: 9.1
Epoch 15: Val Loss: 9.6  ← Current (getting worse)
```

Use `checkpoint_epoch_12.pt` instead of epoch 15.

## Quick Inline Fix (No Restart Needed)

If you can't stop the training, add this cell **while it's running**:

```python
# This won't affect current training but will help next time
import torch

# Modify optimizer on the fly (advanced)
if 'optimizer' in globals():
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001  # Reduce learning rate
        param_group['weight_decay'] = 0.0001  # Increase regularization
    print("✅ Optimizer updated with better settings!")
```

## What to Watch For

After restarting with new config, monitor:

### Good Signs ✅
- Val loss stops increasing
- Val/Train loss ratio decreases (from 1.8 toward 1.2)
- CTC loss improves on validation
- Training is more stable

### Bad Signs ❌
- Val loss still increasing
- Training loss not decreasing
- Model not learning at all (learning rate too low)

## Troubleshooting

### If training is too slow after changes:
```python
# Increase learning rate slightly
learning_rate = 0.0002  # Instead of 0.0001
```

### If still overfitting:
```python
# Increase dropout more
dropout = 0.3  # Instead of 0.2
```

### If underfitting (unlikely):
```python
# Reduce regularization
dropout = 0.15
weight_decay = 0.00001
```

## Summary

**Recommended Action**: Stop at epoch 15, update config, restart with better settings.

**Why**: Your model is memorizing training data. The sooner you fix this, the better your final model will be.

**Time Cost**: ~5 minutes to update and restart vs. potentially wasting hours on an overfit model.
