# Quick Summary: Fine-tuning Changes for Kaggle Notebook

## TL;DR - Minimum Changes Needed

To use `best_model.pt` as a base for fine-tuning, you need to make **3 key changes**:

### 1. Upload Checkpoint to Kaggle
- Upload `best_model (1).pt` as a Kaggle dataset
- Add it as input to your notebook
- Path will be: `/kaggle/input/your-dataset-name/best_model (1).pt`

### 2. Add Checkpoint Loading (1 new cell)
Insert this cell **after Step 2** in your notebook:

```python
import torch
CHECKPOINT_PATH = '/kaggle/input/your-dataset-name/best_model (1).pt'  # UPDATE THIS!

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    torch.save(checkpoint, '/kaggle/working/pretrained_checkpoint.pt')
    RESUME_TRAINING = True
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
else:
    RESUME_TRAINING = False
    print("⚠️  Training from scratch")
```

### 3. Modify Training Parameters
In the **training configuration cell** (around line 679), change:

```python
# BEFORE (training from scratch):
'learning_rate': 0.0001,
'num_epochs': 100,

# AFTER (fine-tuning):
'learning_rate': 0.00001,  # 10x smaller!
'num_epochs': 50,          # Fewer epochs
```

### 4. Add Resume Flag to Training
In the **training execution cell** (around line 1170), add:

```python
# If using command line:
!python /kaggle/working/training_scripts/train_konkanivani_asr.py \
    --train_manifest /kaggle/working/konkani-10k/train_manifest.json \
    --val_manifest /kaggle/working/konkani-10k/val_manifest.json \
    --vocab_file /kaggle/working/custom_vocab.json \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.00001 \
    --ctc_weight 0.9 \
    --resume /kaggle/working/pretrained_checkpoint.pt  # ADD THIS LINE!
```

## That's It!

The training script (`train_konkanivani_asr.py`) already has all the logic to:
- ✅ Load model weights from checkpoint
- ✅ Load optimizer state
- ✅ Resume from the correct epoch
- ✅ Continue improving the model

## Key Parameters for Fine-tuning

| Parameter | From Scratch | Fine-tuning | Why Different? |
|-----------|--------------|-------------|----------------|
| Learning Rate | 0.0001 | 0.00001 | Prevent catastrophic forgetting |
| Num Epochs | 100 | 50 | Model already trained |
| CTC Weight | 0.9 | 0.9 | Keep same (already optimized) |
| Batch Size | 8 | 8 | Keep same |
| Dropout | 0.3 | 0.3-0.4 | Optionally increase to prevent overfitting |

## Expected Behavior

### ✅ Good Signs (Fine-tuning Working)
- Initial validation loss is **lower** than random (around checkpoint's loss)
- Loss decreases **slowly and steadily**
- Model produces **better transcriptions** than before

### ⚠️ Warning Signs (Something Wrong)
- Initial loss is **very high** (weights didn't load)
- Loss **increases** (learning rate too high or overfitting)
- Loss **doesn't change** (learning rate too low)

## File Locations

After training completes:

```
/kaggle/working/
├── checkpoints/
│   ├── best_model.pt          # Your new best model!
│   ├── checkpoint_epoch_5.pt
│   ├── checkpoint_epoch_10.pt
│   └── ...
└── logs/
    └── tensorboard logs
```

## Download Your Fine-tuned Model

```python
from IPython.display import FileLink
FileLink('/kaggle/working/checkpoints/best_model.pt')
```

## Common Issues & Quick Fixes

### Issue: "Checkpoint not found"
**Fix:** Check the path
```python
!ls -lh /kaggle/input/
!ls -lh /kaggle/input/your-dataset-name/
```

### Issue: "RuntimeError: size mismatch"
**Fix:** Vocab size changed. Use the **same vocab.json** from original training
```python
# Check vocab size in checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
print(f"Checkpoint vocab size: {checkpoint['config']['model']['vocab_size']}")
```

### Issue: Loss increases during fine-tuning
**Fix:** Learning rate too high. Try even smaller:
```python
'learning_rate': 0.000005,  # 20x smaller than original
```

### Issue: "CUDA out of memory"
**Fix:** Reduce batch size or use gradient accumulation
```python
'batch_size': 4,                      # Reduce from 8
'gradient_accumulation_steps': 4,    # Increase from 2
```

## Comparison: Before vs After Changes

### BEFORE (Original Notebook)
```python
# Config
config = {
    'training': {
        'learning_rate': 0.0001,
        'num_epochs': 100,
        # ...
    }
}

# Training
!python train_konkanivani_asr.py \
    --learning_rate 0.0001 \
    --num_epochs 100
```

### AFTER (Fine-tuning)
```python
# Load checkpoint
checkpoint = torch.load('/kaggle/input/checkpoint/best_model.pt')
RESUME_TRAINING = True

# Config (adjusted for fine-tuning)
config = {
    'training': {
        'learning_rate': 0.00001,  # Changed!
        'num_epochs': 50,          # Changed!
        # ...
    }
}

# Training (with resume)
!python train_konkanivani_asr.py \
    --learning_rate 0.00001 \
    --num_epochs 50 \
    --resume /kaggle/working/pretrained_checkpoint.pt  # Added!
```

## What Gets Loaded from Checkpoint?

The `best_model.pt` checkpoint contains:

```python
{
    'epoch': 42,                    # Last completed epoch
    'model_state_dict': {...},      # ✅ Model weights (LOADED)
    'optimizer_state_dict': {...},  # ✅ Optimizer state (LOADED)
    'scheduler_state_dict': {...},  # ✅ LR scheduler (LOADED)
    'val_loss': 2.345,             # ✅ Best validation loss (LOADED)
    'config': {...}                # ✅ Original config (REFERENCE)
}
```

## Verification Checklist

Before running fine-tuning, verify:

- [ ] `best_model.pt` uploaded to Kaggle
- [ ] Checkpoint path is correct in code
- [ ] Learning rate reduced to `0.00001`
- [ ] Number of epochs reduced to `50`
- [ ] `--resume` flag added to training command
- [ ] Same `vocab.json` file is being used
- [ ] Checkpoint loads without errors

## Timeline Estimate

Based on your 10.8-hour dataset:

- **From scratch (100 epochs)**: ~30-40 hours on dual T4 GPUs
- **Fine-tuning (50 epochs)**: ~15-20 hours on dual T4 GPUs

**Savings**: ~50% time reduction!

## Next Steps After Fine-tuning

1. **Test the model:**
   ```bash
   python scripts/test_best_model.py \
       --checkpoint checkpoints/best_model.pt \
       --test_manifest konkani-10k/test_manifest.json
   ```

2. **Compare with original:**
   - Original WER: [from first training]
   - Fine-tuned WER: [from this training]
   - Improvement: [calculate difference]

3. **Iterate if needed:**
   - If still not good enough, fine-tune again with even more data
   - Or try different hyperparameters (dropout, model size, etc.)

## Pro Tips

💡 **Save intermediate checkpoints**: The notebook saves every 5 epochs, so you can resume even if it crashes

💡 **Monitor TensorBoard**: Watch validation loss to catch overfitting early

💡 **Keep original checkpoint**: Don't overwrite `best_model.pt` - save fine-tuned version with different name

💡 **Test on held-out data**: Make sure improvements aren't just overfitting to validation set

## Questions?

If something doesn't work:
1. Check the full guide: `FINETUNING_GUIDE.md`
2. Use the ready-made cells: `kaggle_finetuning_cells.md`
3. Verify checkpoint contents: `checkpoint.keys()`
4. Check error messages carefully - they usually tell you what's wrong!
