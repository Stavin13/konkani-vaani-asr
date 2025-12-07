# ASR Training Improvements Guide

## Current Status (Epoch 15/50)
- **Train Loss**: 5.3186 (CTC: 12.1094)
- **Val Loss**: 9.5881 (CTC: 26.2679)
- **Problem**: Significant overfitting (val loss ~2x train loss)

## Applied Changes

### 1. Increased Regularization
- **Dropout**: 0.1 → 0.2 (20% dropout in all layers)
- **Weight Decay**: 0.000001 → 0.0001 (100x stronger L2 regularization)

### 2. Reduced Learning Rate
- **Learning Rate**: 0.0005 → 0.0001 (5x slower)
- **Added**: Cosine annealing schedule with warmup
- **Min LR**: 0.00001 (prevents learning rate from going too low)
- **Warmup**: 1000 steps for stable initial training

### 3. Adjusted Loss Weights
- **CTC Weight**: 0.3 → 0.5 (increased CTC importance)
- **Attention Weight**: 0.7 → 0.5 (balanced hybrid loss)
- **Reason**: CTC helps with better alignment and reduces overfitting

### 4. Added Early Stopping
- **Patience**: 10 epochs without improvement
- **Min Delta**: 0.01 (minimum improvement threshold)
- **Benefit**: Prevents wasting compute on overfit models

### 5. Data Augmentation (Recommended)
Added SpecAugment and other augmentations:
- Time masking (2 masks, max 50 steps each)
- Frequency masking (2 masks, max 10 bins each)
- Gaussian noise injection (0.5%)
- Speed perturbation (0.9x, 1.0x, 1.1x)

## Expected Improvements

### Short Term (Next 5 epochs)
- Val loss should stabilize or decrease
- Gap between train/val loss should narrow
- CTC loss should improve on validation set

### Long Term (Remaining epochs)
- Better generalization to unseen data
- More stable training curves
- Improved Word Error Rate (WER) on test set

## How to Resume Training

```bash
# Make sure you're using the updated config
python training_scripts/train_konkanivani_asr.py \
  --config config/training_config_from_checkpoint15.yaml \
  --resume archives/checkpoint_epoch_15.pt
```

## Monitoring Tips

### Watch These Metrics:
1. **Val/Train Loss Ratio**: Should be < 1.5 (currently ~1.8)
2. **CTC Loss**: Should decrease steadily on both train and val
3. **Gradient Norm**: Should be stable (not exploding)
4. **Learning Rate**: Should decrease gradually with cosine schedule

### When to Stop:
- Early stopping triggers (10 epochs no improvement)
- Val loss starts increasing consistently
- CTC loss plateaus on validation set
- You achieve target WER on test set

## Additional Recommendations

### If Still Overfitting After These Changes:
1. **Increase dropout further** (0.2 → 0.3)
2. **Add label smoothing** (0.1) to attention loss
3. **Reduce model size** (encoder_layers: 12 → 10)
4. **Get more training data** (best solution!)

### If Underfitting (unlikely):
1. Reduce dropout back to 0.1
2. Increase learning rate to 0.0002
3. Increase model capacity (d_model: 256 → 384)

### Data Quality Checks:
```bash
# Verify your data splits
python scripts/validation/inspect_dataset.py

# Check for data leakage
python scripts/validation/fix_data_leakage.py

# Validate audio encoding
python scripts/validation/validate_encoding.py
```

## Implementation Notes

### Data Augmentation Code
You'll need to implement SpecAugment in your data loader:

```python
import torchaudio.transforms as T

class SpecAugment:
    def __init__(self, time_mask_max=50, freq_mask_max=10, 
                 time_mask_num=2, freq_mask_num=2):
        self.time_masking = T.TimeMasking(time_mask_max)
        self.freq_masking = T.FrequencyMasking(freq_mask_max)
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num
    
    def __call__(self, spec):
        for _ in range(self.time_mask_num):
            spec = self.time_masking(spec)
        for _ in range(self.freq_mask_num):
            spec = self.freq_masking(spec)
        return spec
```

### Learning Rate Scheduler
Add to your training loop:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs - warmup_epochs,
    eta_min=config['training']['min_lr']
)
```

## Results Tracking

Create a tracking sheet with these columns:
- Epoch
- Train Loss (Total)
- Train Loss (CTC)
- Val Loss (Total)
- Val Loss (CTC)
- Learning Rate
- Time per Epoch
- Notes

## Questions?

If you see:
- **Loss exploding**: Reduce learning rate further
- **No improvement**: Check data quality and augmentation
- **Slow convergence**: Increase batch size or learning rate slightly

Good luck with your training! 🚀
