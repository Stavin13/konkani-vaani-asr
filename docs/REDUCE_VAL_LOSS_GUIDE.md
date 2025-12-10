# Guide: Getting Validation Loss Below 9

## Current Situation
- **Best validation loss achieved**: 9.47 (at Epoch 4)
- **Problem**: Model overfitting - training loss improves but validation loss plateaus
- **Goal**: Get validation loss below 9.0

## ✅ Solution: Stronger Regularization

The training script has been updated with new parameters to control regularization:
- `--weight_decay`: L2 regularization strength
- `--dropout`: Dropout rate for the model
- `--ctc_weight`: Balance between CTC and attention loss
- `--save_every`: How often to save checkpoints

## 🎯 Recommended Training Command

Use this in your Kaggle notebook:

```python
print("="*70)
print("🎯 TRAINING WITH STRONGER REGULARIZATION")
print("="*70)
print(f"Batch size: {BATCH_SIZE}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print("="*70)

import os
os.environ['PYTHONPATH'] = os.getcwd()

!PYTHONPATH={os.getcwd()} python3 training_scripts/train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file data/vocab.json \
    --resume checkpoints/checkpoint_epoch_5.pt \
    --batch_size {BATCH_SIZE} \
    --gradient_accumulation_steps {GRAD_ACCUM} \
    --num_epochs 50 \
    --learning_rate 0.00005 \
    --weight_decay 0.0005 \
    --dropout 0.3 \
    --ctc_weight 0.6 \
    --save_every 2 \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6 \
    --mixed_precision \
    --checkpoint_dir checkpoints \
    --log_dir logs
```

## 📊 What Changed vs Previous Training

| Parameter | Previous | New | Why |
|-----------|----------|-----|-----|
| Learning Rate | 0.0001 | 0.00005 | Slower, more careful updates |
| Weight Decay | 0.000001 | 0.0005 | 500x stronger L2 regularization |
| Dropout | 0.2 | 0.3 | More aggressive dropout |
| CTC Weight | 0.3 | 0.6 | CTC loss helps generalization |
| Save Every | 5 | 2 | Catch best model faster |

## 🎯 Expected Results

**Timeline:**
- **Epochs 1-5**: Val loss should stabilize around 9.3-9.5
- **Epochs 6-15**: Gradual improvement to 8.8-9.2
- **Epochs 16-25**: Target range 8.5-9.0

**Signs of Success:**
- ✅ Val loss decreasing steadily
- ✅ Smaller gap between train and val loss
- ✅ No sudden loss spikes
- ✅ CTC loss improving on validation set

**If Val Loss Still Plateaus:**
Try even stronger regularization:
```bash
--learning_rate 0.00003
--weight_decay 0.001
--dropout 0.4
```

## 🔧 Alternative: Start from Scratch with Strong Regularization

If you don't have checkpoint_epoch_5.pt on Kaggle, remove the `--resume` line:

```python
!PYTHONPATH={os.getcwd()} python3 training_scripts/train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file data/vocab.json \
    --batch_size {BATCH_SIZE} \
    --gradient_accumulation_steps {GRAD_ACCUM} \
    --num_epochs 50 \
    --learning_rate 0.00005 \
    --weight_decay 0.0005 \
    --dropout 0.3 \
    --ctc_weight 0.6 \
    --save_every 2 \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6 \
    --mixed_precision \
    --checkpoint_dir checkpoints \
    --log_dir logs
```

This will take longer (need to train from epoch 1) but should reach val loss < 9 by epoch 20-25.

## 📈 Monitoring Progress

Watch for these metrics in the logs:
```
Epoch X/50
Train Loss: X.XX (CTC: XX.XX)
Val Loss: X.XX (CTC: XX.XX)
✅ Saved best model with val_loss: X.XX
```

**Good progress looks like:**
- Train loss: 6.5 → 5.8 → 5.2 → 4.8
- Val loss: 9.5 → 9.2 → 8.9 → 8.6
- Gap narrowing over time

## 🚨 Troubleshooting

**If val loss increases:**
- Learning rate too high → reduce to 0.00003
- Not enough regularization → increase dropout to 0.4

**If training is too slow:**
- Increase batch size if you have GPU memory
- Reduce gradient accumulation steps

**If you see loss spikes (e.g., 99.96, 133.99):**
- This is normal early in training
- Should disappear after epoch 5-10
- If persists, reduce learning rate
