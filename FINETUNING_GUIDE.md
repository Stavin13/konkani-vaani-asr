# Fine-tuning Guide: Using best_model.pt as Base

## Overview
You have a pre-trained model checkpoint (`best_model.pt`) that you want to use as a starting point for further training to improve your ASR model. This guide explains the changes needed in your Kaggle notebook.

## What's in best_model.pt?

Based on the training script, your checkpoint contains:
```python
{
    'epoch': <epoch_number>,
    'model_state_dict': <model_weights>,
    'optimizer_state_dict': <optimizer_state>,
    'scheduler_state_dict': <scheduler_state>,
    'val_loss': <validation_loss>,
    'config': <training_config>
}
```

## Changes Required in Kaggle Notebook

### 1. Upload best_model.pt to Kaggle

**Option A: As a Dataset**
1. Create a new Kaggle dataset with `best_model.pt`
2. Add it as input to your notebook
3. Reference it in the notebook

**Option B: Direct Upload**
1. Upload to notebook's input section
2. Access from `/kaggle/input/`

### 2. Modify the Notebook - Add Checkpoint Loading Section

Add this **AFTER** the model creation (around line 940 in the notebook, after the imports are verified):

```python
# ============================================================
# FINE-TUNING: Load Pre-trained Checkpoint
# ============================================================

# Path to your uploaded checkpoint
CHECKPOINT_PATH = '/kaggle/input/your-checkpoint-dataset/best_model.pt'  # UPDATE THIS PATH

# Check if checkpoint exists
import os
if os.path.exists(CHECKPOINT_PATH):
    print(f"🔄 Loading pre-trained checkpoint from: {CHECKPOINT_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Display checkpoint info
    print(f"\n📊 Checkpoint Information:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # Extract config from checkpoint
    checkpoint_config = checkpoint.get('config', {})
    print(f"\n🔧 Original Training Config:")
    print(f"  CTC Weight: {checkpoint_config.get('ctc_weight', 'N/A')}")
    print(f"  Learning Rate: {checkpoint_config.get('learning_rate', 'N/A')}")
    
    # Store for later use
    PRETRAINED_CHECKPOINT = checkpoint
    RESUME_TRAINING = True
    print("\n✅ Checkpoint loaded successfully!")
    
else:
    print(f"⚠️  Checkpoint not found at: {CHECKPOINT_PATH}")
    print("   Training from scratch...")
    RESUME_TRAINING = False
    PRETRAINED_CHECKPOINT = None
```

### 3. Modify Training Configuration

**IMPORTANT**: For fine-tuning, you should use a **lower learning rate** than the original training. Add this after the config creation (around line 679):

```python
# Training configuration with FIXES
import yaml

# Base configuration
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
        # 🔥 FINE-TUNING: Use lower learning rate (1/10th of original)
        'learning_rate': 0.00001,      # Was 0.0001, now 10x smaller for fine-tuning
        'weight_decay': 0.0001,
        'grad_clip': 5.0,
        'ctc_weight': 0.9,             # Keep the same CTC weight
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'num_epochs': 50,              # 🔥 Fewer epochs for fine-tuning
        'save_every': 5,
        'test_every': 5
    },
    'data': {
        'train_manifest': str(manifest_dir / 'train_manifest.json') if (manifest_dir / 'train_manifest.json').exists() else str(manifest_dir / 'train.json'),
        'val_manifest': str(manifest_dir / 'val_manifest.json') if (manifest_dir / 'val_manifest.json').exists() else str(manifest_dir / 'val.json'),
        'vocab_file': str(manifest_dir / 'vocab.json') if (manifest_dir / 'vocab.json').exists() else '/kaggle/working/konkani-10k/vocab.json',
        'num_workers': 2
    },
    'paths': {
        'checkpoint_dir': '/kaggle/working/checkpoints',
        'log_dir': '/kaggle/working/logs'
    },
    'device': 'cuda',
    # 🔥 FINE-TUNING: Add resume flag
    'resume_from': CHECKPOINT_PATH if RESUME_TRAINING else None
}

# Save config
os.makedirs('/kaggle/working/config', exist_ok=True)
with open('/kaggle/working/config/training_config_finetuned.yaml', 'w') as f:
    yaml.dump(config, f)

print("✓ Training config saved for FINE-TUNING:")
print(f"  - Learning rate: {config['training']['learning_rate']} (10x smaller for fine-tuning)")
print(f"  - CTC weight: {config['training']['ctc_weight']}")
print(f"  - Num epochs: {config['training']['num_epochs']}")
print(f"  - Resume from: {config.get('resume_from', 'None')}")
```

### 4. Modify the Training Script Execution

The current notebook runs the training script directly. You need to pass the `--resume` argument. Find the cell that runs training (around line 1170) and modify it:

**BEFORE:**
```python
!python /kaggle/working/training_scripts/train_konkanivani_asr.py \
    --train_manifest /kaggle/working/konkani-10k/train_manifest.json \
    --val_manifest /kaggle/working/konkani-10k/val_manifest.json \
    --vocab_file /kaggle/working/custom_vocab.json \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 0.0001 \
    --ctc_weight 0.9
```

**AFTER:**
```python
# Build training command
train_cmd = [
    "python", "/kaggle/working/training_scripts/train_konkanivani_asr.py",
    "--train_manifest", "/kaggle/working/konkani-10k/train_manifest.json",
    "--val_manifest", "/kaggle/working/konkani-10k/val_manifest.json",
    "--vocab_file", "/kaggle/working/custom_vocab.json",
    "--batch_size", "8",
    "--num_epochs", "50",  # 🔥 Fewer epochs for fine-tuning
    "--learning_rate", "0.00001",  # 🔥 Lower learning rate for fine-tuning
    "--ctc_weight", "0.9",
    "--gradient_accumulation_steps", "2",
    "--mixed_precision",
    "--checkpoint_dir", "/kaggle/working/checkpoints",
    "--log_dir", "/kaggle/working/logs"
]

# 🔥 FINE-TUNING: Add resume argument if checkpoint exists
if RESUME_TRAINING and PRETRAINED_CHECKPOINT is not None:
    # First, save the checkpoint to working directory
    checkpoint_path = '/kaggle/working/pretrained_checkpoint.pt'
    torch.save(PRETRAINED_CHECKPOINT, checkpoint_path)
    print(f"💾 Saved checkpoint to: {checkpoint_path}")
    
    # Add resume argument
    train_cmd.extend(["--resume", checkpoint_path])
    print("🔄 Training will resume from checkpoint")

# Run training
import subprocess
result = subprocess.run(train_cmd, capture_output=False, text=True)
```

### 5. Alternative: Inline Training (Recommended for Fine-tuning)

Instead of calling the training script, you can load the model directly in the notebook for more control:

```python
# ============================================================
# INLINE FINE-TUNING (More Control)
# ============================================================

# Add working directory to path
import sys
sys.path.insert(0, '/kaggle/working')

from models.konkanivani_asr import create_konkanivani_model
from data.audio_processing.dataset import create_dataloaders
from data.audio_processing.text_tokenizer import KonkaniTokenizer
from training_scripts.train_konkanivani_asr import ASRTrainer

# Load tokenizer
tokenizer = KonkaniTokenizer('/kaggle/working/custom_vocab.json')
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    '/kaggle/working/konkani-10k/train_manifest.json',
    '/kaggle/working/konkani-10k/val_manifest.json',
    tokenizer,
    batch_size=8,
    num_workers=2
)

# Create model
model_config = {
    'input_dim': 80,
    'd_model': 128,
    'encoder_layers': 8,
    'decoder_layers': 6,
    'num_heads': 4,
    'conv_kernel_size': 31,
    'dropout': 0.3
}
model = create_konkanivani_model(vocab_size=tokenizer.vocab_size, config=model_config)

# 🔥 FINE-TUNING: Load pre-trained weights
if RESUME_TRAINING and PRETRAINED_CHECKPOINT is not None:
    print("\n🔄 Loading pre-trained model weights...")
    model.load_state_dict(PRETRAINED_CHECKPOINT['model_state_dict'])
    print("✅ Pre-trained weights loaded!")

# Setup device (multi-GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Training config for fine-tuning
training_config = {
    'learning_rate': 0.00001,  # 🔥 Lower LR for fine-tuning
    'weight_decay': 0.0001,
    'grad_clip': 5.0,
    'ctc_weight': 0.9,
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'log_dir': '/kaggle/working/logs',
    'save_every': 5,
    'mixed_precision': True,
    'gradient_accumulation_steps': 2
}

# Create trainer
trainer = ASRTrainer(model, tokenizer, train_loader, val_loader, device, training_config)

# 🔥 FINE-TUNING: Optionally load optimizer state too (for warm restart)
if RESUME_TRAINING and PRETRAINED_CHECKPOINT is not None:
    # If you want to continue with the same optimizer state:
    # trainer.optimizer.load_state_dict(PRETRAINED_CHECKPOINT['optimizer_state_dict'])
    # trainer.scheduler.load_state_dict(PRETRAINED_CHECKPOINT['scheduler_state_dict'])
    
    # OR reset optimizer for fresh fine-tuning (recommended)
    print("🔄 Using fresh optimizer for fine-tuning")

# Start fine-tuning
print("\n🚀 Starting fine-tuning...")
trainer.train(num_epochs=50)
```

## Key Fine-tuning Recommendations

### Learning Rate Strategy
- **Original training**: `0.0001`
- **Fine-tuning**: `0.00001` (10x smaller)
- **Why**: Prevents catastrophic forgetting of learned features

### Number of Epochs
- **Original training**: `100` epochs
- **Fine-tuning**: `20-50` epochs
- **Why**: Model is already trained, needs fewer epochs to adapt

### What to Keep vs. Change

**KEEP THE SAME:**
- ✅ Model architecture (d_model, layers, etc.)
- ✅ Vocabulary size
- ✅ CTC weight (0.9)
- ✅ Batch size
- ✅ Gradient clipping

**CHANGE FOR FINE-TUNING:**
- 🔄 Learning rate (reduce by 10x)
- 🔄 Number of epochs (reduce to 20-50)
- 🔄 Optionally: Increase dropout slightly (0.3 → 0.4) to prevent overfitting

### Monitoring Fine-tuning

Watch for these signs:
- ✅ **Good**: Validation loss continues to decrease
- ✅ **Good**: Training loss decreases slowly and steadily
- ⚠️ **Warning**: Validation loss increases (overfitting)
- ⚠️ **Warning**: Training loss decreases too fast (learning rate too high)

## Complete Workflow

1. **Upload checkpoint** to Kaggle as dataset
2. **Add checkpoint loading** section to notebook
3. **Modify config** with lower learning rate
4. **Run fine-tuning** with `--resume` flag
5. **Monitor progress** in TensorBoard logs
6. **Save best model** when validation loss improves

## Example: Full Modified Notebook Cell

Here's a complete cell you can add to your notebook:

```python
# ============================================================
# FINE-TUNING SETUP
# ============================================================

# 1. Set checkpoint path (update this!)
CHECKPOINT_PATH = '/kaggle/input/konkani-asr-checkpoint/best_model.pt'

# 2. Load checkpoint
if os.path.exists(CHECKPOINT_PATH):
    print("🔄 Loading pre-trained checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Save to working directory for training script
    working_checkpoint = '/kaggle/working/pretrained_model.pt'
    torch.save(checkpoint, working_checkpoint)
    
    RESUME_FROM = working_checkpoint
    print(f"✅ Checkpoint ready at: {RESUME_FROM}")
else:
    RESUME_FROM = None
    print("⚠️  No checkpoint found, training from scratch")

# 3. Run training with resume
!python /kaggle/working/training_scripts/train_konkanivani_asr.py \
    --train_manifest /kaggle/working/konkani-10k/train_manifest.json \
    --val_manifest /kaggle/working/konkani-10k/val_manifest.json \
    --vocab_file /kaggle/working/custom_vocab.json \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 0.00001 \
    --ctc_weight 0.9 \
    --gradient_accumulation_steps 2 \
    --mixed_precision \
    --checkpoint_dir /kaggle/working/checkpoints \
    --log_dir /kaggle/working/logs \
    --resume {RESUME_FROM if RESUME_FROM else ""}
```

## Troubleshooting

### Issue: "RuntimeError: size mismatch"
**Cause**: Vocabulary size changed between original training and fine-tuning
**Solution**: Ensure you use the **same vocab.json** file that was used during original training

### Issue: "KeyError: 'model_state_dict'"
**Cause**: Checkpoint format is different
**Solution**: Check checkpoint contents with `checkpoint.keys()`

### Issue: Loss increases during fine-tuning
**Cause**: Learning rate too high
**Solution**: Reduce learning rate further (try 0.000005)

### Issue: No improvement in validation loss
**Cause**: Model already converged or data is too similar
**Solution**: 
- Try different data augmentation
- Increase model capacity
- Check if new data is significantly different

## Summary

**Minimum changes needed:**
1. Upload `best_model.pt` to Kaggle
2. Add `--resume /path/to/best_model.pt` to training command
3. Change `--learning_rate` from `0.0001` to `0.00001`
4. Change `--num_epochs` from `100` to `50`

That's it! The training script already has all the logic to resume from checkpoints.
