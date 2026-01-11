# Side-by-Side: Notebook Changes for Fine-tuning

This document shows **exactly** what to change in your Kaggle notebook to enable fine-tuning.

---

## Change 1: Add Checkpoint Loading Cell

**Location:** Insert NEW cell after Step 2 (after dataset verification)

**What to add:**

```python
# ============================================================
# NEW CELL: Load Pre-trained Checkpoint for Fine-tuning
# ============================================================

import torch
import os

# UPDATE THIS PATH to your uploaded checkpoint!
CHECKPOINT_PATH = '/kaggle/input/your-checkpoint-dataset/best_model (1).pt'

if os.path.exists(CHECKPOINT_PATH):
    print(f"🔄 Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Save to working directory
    torch.save(checkpoint, '/kaggle/working/pretrained_checkpoint.pt')
    
    RESUME_TRAINING = True
    print(f"✅ Loaded! Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
else:
    RESUME_TRAINING = False
    print("⚠️  No checkpoint found, training from scratch")
```

---

## Change 2: Modify Training Configuration

**Location:** Cell around line 679 (training config)

### BEFORE (Original):
```python
config = {
    'model': {
        'vocab_size': 82,
        'input_dim': 80,
        'd_model': 128,
        'encoder_layers': 8,
        'decoder_layers': 6,
        'num_heads': 4,
        'conv_kernel_size': 31,
        'dropout': 0.3
    },
    'training': {
        'learning_rate': 0.0001,      # ← CHANGE THIS
        'weight_decay': 0.0001,
        'grad_clip': 5.0,
        'ctc_weight': 0.9,
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'num_epochs': 100,            # ← CHANGE THIS
        'save_every': 5,
        'test_every': 5
    },
    # ... rest of config
}
```

### AFTER (Fine-tuning):
```python
# Adjust parameters based on whether we're fine-tuning
if RESUME_TRAINING:
    learning_rate = 0.00001  # 10x smaller for fine-tuning
    num_epochs = 50          # Fewer epochs
else:
    learning_rate = 0.0001   # Normal for training from scratch
    num_epochs = 100

config = {
    'model': {
        'vocab_size': 82,
        'input_dim': 80,
        'd_model': 128,
        'encoder_layers': 8,
        'decoder_layers': 6,
        'num_heads': 4,
        'conv_kernel_size': 31,
        'dropout': 0.3
    },
    'training': {
        'learning_rate': learning_rate,     # ✅ CHANGED
        'weight_decay': 0.0001,
        'grad_clip': 5.0,
        'ctc_weight': 0.9,
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'num_epochs': num_epochs,           # ✅ CHANGED
        'save_every': 5,
        'test_every': 5
    },
    # ... rest of config
}

print(f"Mode: {'Fine-tuning' if RESUME_TRAINING else 'From scratch'}")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {num_epochs}")
```

---

## Change 3: Modify Training Execution

**Location:** Cell around line 1170 (where training starts)

### BEFORE (Original):
```python
# The notebook currently has inline training code
# that doesn't load checkpoints
```

### AFTER (Fine-tuning):

**Option A: Using Training Script (Simpler)**

```python
# Build command with conditional resume
cmd = f"""python /kaggle/working/training_scripts/train_konkanivani_asr.py \
    --train_manifest /kaggle/working/konkani-10k/train_manifest.json \
    --val_manifest /kaggle/working/konkani-10k/val_manifest.json \
    --vocab_file /kaggle/working/custom_vocab.json \
    --batch_size 8 \
    --num_epochs {config['training']['num_epochs']} \
    --learning_rate {config['training']['learning_rate']} \
    --ctc_weight 0.9 \
    --gradient_accumulation_steps 2 \
    --mixed_precision \
    --checkpoint_dir /kaggle/working/checkpoints \
    --log_dir /kaggle/working/logs"""

# Add resume flag if fine-tuning
if RESUME_TRAINING:
    cmd += " --resume /kaggle/working/pretrained_checkpoint.pt"

# Run training
!{cmd}
```

**Option B: Inline Training (More Control)**

```python
import sys
sys.path.insert(0, '/kaggle/working')

from models.konkanivani_asr import create_konkanivani_model
from data.audio_processing.dataset import create_dataloaders
from data.audio_processing.text_tokenizer import KonkaniTokenizer
from training_scripts.train_konkanivani_asr import ASRTrainer
import torch.nn as nn

# Load tokenizer
tokenizer = KonkaniTokenizer(config['data']['vocab_file'])

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    config['data']['train_manifest'],
    config['data']['val_manifest'],
    tokenizer,
    batch_size=config['training']['batch_size'],
    num_workers=2
)

# Create model
model = create_konkanivani_model(
    vocab_size=tokenizer.vocab_size,
    config=config['model']
)

# ✅ LOAD PRE-TRAINED WEIGHTS IF FINE-TUNING
if RESUME_TRAINING:
    checkpoint = torch.load('/kaggle/working/pretrained_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded pre-trained weights from epoch {checkpoint['epoch']}")

# Setup device and multi-GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Create trainer
trainer = ASRTrainer(
    model=model,
    tokenizer=tokenizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    config=config['training']
)

# Start training
trainer.train(num_epochs=config['training']['num_epochs'])
```

---

## Summary of Changes

| What | Where | Change |
|------|-------|--------|
| **Add checkpoint loading** | After Step 2 | NEW cell to load `best_model.pt` |
| **Learning rate** | Config cell (~line 679) | `0.0001` → `0.00001` |
| **Number of epochs** | Config cell (~line 679) | `100` → `50` |
| **Load model weights** | Training cell (~line 1170) | Add `model.load_state_dict()` or `--resume` flag |

---

## Minimal Changes (If You're in a Hurry)

If you just want to get it working ASAP, make these **2 changes**:

### 1. Add this cell after Step 2:
```python
import torch
checkpoint = torch.load('/kaggle/input/your-dataset/best_model (1).pt', map_location='cpu')
torch.save(checkpoint, '/kaggle/working/pretrained_checkpoint.pt')
RESUME_TRAINING = True
```

### 2. Change the training command to:
```python
!python /kaggle/working/training_scripts/train_konkanivani_asr.py \
    --train_manifest /kaggle/working/konkani-10k/train_manifest.json \
    --val_manifest /kaggle/working/konkani-10k/val_manifest.json \
    --vocab_file /kaggle/working/custom_vocab.json \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.00001 \
    --ctc_weight 0.9 \
    --resume /kaggle/working/pretrained_checkpoint.pt
```

**That's it!** The training script handles everything else automatically.

---

## Verification: How to Know It's Working

### ✅ Checkpoint Loaded Successfully
You should see:
```
🔄 Loading checkpoint from: /kaggle/input/.../best_model.pt
📂 Loading checkpoint from: /kaggle/working/pretrained_checkpoint.pt
✅ Resumed from epoch 42
   Best val loss so far: 2.345
```

### ✅ Training Starts from Pre-trained Weights
First epoch validation loss should be **close to checkpoint's loss**, not random:
```
Epoch 1/50
  Train Loss: 2.234  ← Should be low, not ~10+
  Val Loss: 2.189    ← Should be close to checkpoint's 2.345
```

### ❌ Something Wrong
If you see:
```
Epoch 1/50
  Train Loss: 12.456  ← Very high = weights didn't load!
  Val Loss: 11.234
```

Then checkpoint didn't load. Check:
1. Path is correct
2. Vocab size matches
3. No error messages during loading

---

## Testing Your Changes Locally First

Before running on Kaggle (which costs GPU time), test locally:

```python
# Test checkpoint loading
import torch
checkpoint = torch.load('best_model (1).pt', map_location='cpu')
print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val loss: {checkpoint['val_loss']}")
print(f"Vocab size: {checkpoint['config']['model']['vocab_size']}")

# Test model loading
from models.konkanivani_asr import create_konkanivani_model
model = create_konkanivani_model(vocab_size=82, config={...})
model.load_state_dict(checkpoint['model_state_dict'])
print("✅ Model weights loaded successfully!")
```

---

## What NOT to Change

Keep these the **same** as original training:

- ❌ Model architecture (`d_model`, `encoder_layers`, etc.)
- ❌ Vocabulary file (must use same `vocab.json`)
- ❌ CTC weight (already optimized at 0.9)
- ❌ Batch size (unless you have memory issues)
- ❌ Gradient clipping (5.0 is good)

Only change:
- ✅ Learning rate (make it smaller)
- ✅ Number of epochs (make it fewer)
- ✅ Optionally: Dropout (can increase slightly)

---

## Final Checklist

Before running the notebook:

- [ ] Uploaded `best_model (1).pt` to Kaggle
- [ ] Updated `CHECKPOINT_PATH` in code
- [ ] Added checkpoint loading cell
- [ ] Changed learning rate to `0.00001`
- [ ] Changed epochs to `50`
- [ ] Added `--resume` flag or `load_state_dict()` call
- [ ] Verified vocab file is the same
- [ ] Checked that checkpoint loads without errors

If all checked, you're ready to run! 🚀
