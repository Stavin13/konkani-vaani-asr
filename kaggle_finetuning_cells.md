# Kaggle Notebook Cells for Fine-tuning

Copy and paste these cells into your Kaggle notebook to enable fine-tuning from `best_model.pt`.

## Cell 1: Upload and Load Checkpoint (Insert after Step 2)

```python
# ============================================================
# FINE-TUNING: Load Pre-trained Checkpoint
# ============================================================

import os
import torch

# UPDATE THIS PATH to match your uploaded checkpoint dataset
# Option 1: If uploaded as Kaggle dataset
CHECKPOINT_PATH = '/kaggle/input/konkani-asr-checkpoint/best_model (1).pt'

# Option 2: If uploaded directly to notebook
# CHECKPOINT_PATH = '/kaggle/input/best_model.pt'

# Check if checkpoint exists
RESUME_TRAINING = False
PRETRAINED_CHECKPOINT = None

if os.path.exists(CHECKPOINT_PATH):
    print(f"🔄 Loading pre-trained checkpoint from: {CHECKPOINT_PATH}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        # Display checkpoint info
        print(f"\n📊 Checkpoint Information:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        # Extract config from checkpoint
        checkpoint_config = checkpoint.get('config', {})
        if checkpoint_config:
            print(f"\n🔧 Original Training Config:")
            print(f"  CTC Weight: {checkpoint_config.get('ctc_weight', 'N/A')}")
            print(f"  Learning Rate: {checkpoint_config.get('learning_rate', 'N/A')}")
            print(f"  Batch Size: {checkpoint_config.get('batch_size', 'N/A')}")
        
        # Check what's in the checkpoint
        print(f"\n📦 Checkpoint Contents:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        # Store for later use
        PRETRAINED_CHECKPOINT = checkpoint
        RESUME_TRAINING = True
        
        # Copy to working directory for training script
        working_checkpoint_path = '/kaggle/working/pretrained_checkpoint.pt'
        torch.save(checkpoint, working_checkpoint_path)
        print(f"\n💾 Checkpoint copied to: {working_checkpoint_path}")
        
        print("\n✅ Checkpoint loaded successfully! Will resume training from this point.")
        
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        print("   Will train from scratch instead.")
        RESUME_TRAINING = False
        PRETRAINED_CHECKPOINT = None
        
else:
    print(f"⚠️  Checkpoint not found at: {CHECKPOINT_PATH}")
    print("   Available input files:")
    !ls -lh /kaggle/input/
    print("\n   Training from scratch...")
    RESUME_TRAINING = False
    PRETRAINED_CHECKPOINT = None
```

## Cell 2: Modified Training Configuration (Replace existing config cell around line 679)

```python
# Training configuration with FINE-TUNING settings
import yaml

# Determine learning rate based on whether we're fine-tuning
if RESUME_TRAINING:
    # Fine-tuning: Use 10x smaller learning rate
    learning_rate = 0.00001
    num_epochs = 50  # Fewer epochs for fine-tuning
    print("🔄 FINE-TUNING MODE")
    print(f"  Using reduced learning rate: {learning_rate}")
    print(f"  Using fewer epochs: {num_epochs}")
else:
    # Training from scratch: Use normal learning rate
    learning_rate = 0.0001
    num_epochs = 100
    print("🆕 TRAINING FROM SCRATCH")
    print(f"  Using normal learning rate: {learning_rate}")
    print(f"  Using full epochs: {num_epochs}")

config = {
    'model': {
        'vocab_size': 82,  # Will be updated after vocab generation
        'input_dim': 80,
        'd_model': 128,
        'encoder_layers': 8,
        'decoder_layers': 6,
        'num_heads': 4,
        'conv_kernel_size': 31,
        'dropout': 0.3
    },
    'training': {
        'learning_rate': learning_rate,
        'weight_decay': 0.0001,
        'grad_clip': 5.0,
        'ctc_weight': 0.9,
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'num_epochs': num_epochs,
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
    'device': 'cuda'
}

# Save config
os.makedirs('/kaggle/working/config', exist_ok=True)
config_filename = 'training_config_finetuned.yaml' if RESUME_TRAINING else 'training_config_fixed.yaml'
with open(f'/kaggle/working/config/{config_filename}', 'w') as f:
    yaml.dump(config, f)

print(f"\n✓ Training config saved: {config_filename}")
print(f"  - CTC weight: {config['training']['ctc_weight']}")
print(f"  - Learning rate: {config['training']['learning_rate']}")
print(f"  - Gradient clip: {config['training']['grad_clip']}")
print(f"  - Testing: Every {config['training']['test_every']} epochs")
print(f"  - Total epochs: {config['training']['num_epochs']}")
```

## Cell 3: Modified Training Execution (Replace the training cell around line 1170)

```python
# ============================================================
# Start Training (with optional fine-tuning)
# ============================================================

import sys
sys.path.insert(0, '/kaggle/working')

# Import required modules
from models.konkanivani_asr import create_konkanivani_model
from data.audio_processing.dataset import create_dataloaders
from data.audio_processing.text_tokenizer import KonkaniTokenizer
from training_scripts.train_konkanivani_asr import ASRTrainer
import torch.nn as nn

print("="*60)
if RESUME_TRAINING:
    print("🔄 FINE-TUNING MODE")
    print(f"   Resuming from checkpoint: {CHECKPOINT_PATH}")
else:
    print("🆕 TRAINING FROM SCRATCH")
print("="*60)

# Load tokenizer
tokenizer = KonkaniTokenizer(config['data']['vocab_file'])
print(f"\n📚 Vocabulary size: {tokenizer.vocab_size}")

# Update model config with actual vocab size
config['model']['vocab_size'] = tokenizer.vocab_size

# Create dataloaders
print(f"\n📊 Creating dataloaders...")
train_loader, val_loader = create_dataloaders(
    config['data']['train_manifest'],
    config['data']['val_manifest'],
    tokenizer,
    batch_size=config['training']['batch_size'],
    num_workers=config['data']['num_workers']
)
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# Create model
print(f"\n🏗️  Creating model...")
model = create_konkanivani_model(
    vocab_size=tokenizer.vocab_size, 
    config=config['model']
)

# Count parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model parameters: {num_params:,}")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f"\n🖥️  Device setup:")
print(f"  Device: {device}")
print(f"  GPUs available: {num_gpus}")

# Load pre-trained weights if fine-tuning
if RESUME_TRAINING and PRETRAINED_CHECKPOINT is not None:
    print(f"\n🔄 Loading pre-trained model weights...")
    try:
        # Load model state dict
        model.load_state_dict(PRETRAINED_CHECKPOINT['model_state_dict'])
        print("  ✅ Model weights loaded successfully!")
        
        # Get starting epoch
        start_epoch = PRETRAINED_CHECKPOINT.get('epoch', 0)
        best_val_loss = PRETRAINED_CHECKPOINT.get('val_loss', float('inf'))
        print(f"  📊 Previous training:")
        print(f"     - Completed epochs: {start_epoch}")
        print(f"     - Best val loss: {best_val_loss:.4f}")
        
    except Exception as e:
        print(f"  ❌ Error loading weights: {e}")
        print("  ⚠️  Will train from scratch instead")
        RESUME_TRAINING = False

# Multi-GPU setup
if num_gpus > 1:
    print(f"\n🚀 Enabling multi-GPU training ({num_gpus} GPUs)")
    model = nn.DataParallel(model)
    effective_batch_size = config['training']['batch_size'] * num_gpus * config['training']['gradient_accumulation_steps']
    print(f"  Batch size per GPU: {config['training']['batch_size']}")
    print(f"  Total batch per step: {config['training']['batch_size'] * num_gpus}")
    print(f"  Effective batch size: {effective_batch_size}")

# Create trainer
print(f"\n🎯 Creating trainer...")
trainer = ASRTrainer(
    model=model,
    tokenizer=tokenizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    config=config['training']
)

# Optionally load optimizer state for warm restart
# (Comment out if you want fresh optimizer for fine-tuning)
if RESUME_TRAINING and PRETRAINED_CHECKPOINT is not None:
    # Option 1: Load optimizer state (warm restart - continues exactly where left off)
    # try:
    #     trainer.optimizer.load_state_dict(PRETRAINED_CHECKPOINT['optimizer_state_dict'])
    #     trainer.scheduler.load_state_dict(PRETRAINED_CHECKPOINT['scheduler_state_dict'])
    #     trainer.best_val_loss = PRETRAINED_CHECKPOINT.get('val_loss', float('inf'))
    #     print("  ✅ Optimizer state loaded (warm restart)")
    # except:
    #     print("  ⚠️  Could not load optimizer state, using fresh optimizer")
    
    # Option 2: Use fresh optimizer (recommended for fine-tuning)
    print("  🔄 Using fresh optimizer for fine-tuning (recommended)")
    trainer.best_val_loss = PRETRAINED_CHECKPOINT.get('val_loss', float('inf'))

# Start training
print(f"\n{'='*60}")
print(f"🚀 STARTING TRAINING")
print(f"{'='*60}")
print(f"  Mode: {'Fine-tuning' if RESUME_TRAINING else 'From scratch'}")
print(f"  Learning rate: {config['training']['learning_rate']}")
print(f"  Epochs: {config['training']['num_epochs']}")
print(f"  CTC weight: {config['training']['ctc_weight']}")
print(f"  Checkpoints: {config['paths']['checkpoint_dir']}")
print(f"{'='*60}\n")

# Train!
trainer.train(num_epochs=config['training']['num_epochs'])

print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"  Best model saved to: {config['paths']['checkpoint_dir']}/best_model.pt")
print(f"  Logs saved to: {config['paths']['log_dir']}")
```

## Quick Start Instructions

1. **Upload your checkpoint:**
   - Go to Kaggle notebook
   - Click "Add Data" → "Upload" → Select `best_model (1).pt`
   - OR create a dataset with the checkpoint and add it as input

2. **Update the checkpoint path:**
   - In Cell 1, change `CHECKPOINT_PATH` to match where you uploaded the file
   - Example: `/kaggle/input/my-checkpoint/best_model (1).pt`

3. **Insert the cells:**
   - Insert Cell 1 after your dataset verification (after Step 2)
   - Replace the existing config cell with Cell 2
   - Replace the existing training cell with Cell 3

4. **Run the notebook:**
   - The notebook will automatically detect the checkpoint
   - It will use fine-tuning settings (lower LR, fewer epochs)
   - Training will resume from the pre-trained weights

## What Happens Automatically

✅ **Checkpoint detected** → Fine-tuning mode activated
- Learning rate: `0.00001` (10x smaller)
- Epochs: `50` (instead of 100)
- Model weights loaded from checkpoint
- Fresh optimizer (prevents overfitting)

❌ **No checkpoint** → Normal training mode
- Learning rate: `0.0001` (normal)
- Epochs: `100` (full training)
- Model initialized randomly
- Standard training from scratch

## Monitoring Progress

Check these during training:

1. **First few epochs:**
   - Loss should be lower than random initialization
   - Should start around the checkpoint's validation loss

2. **Training progress:**
   - Validation loss should gradually decrease
   - If it increases, learning rate might be too high

3. **Checkpoints:**
   - Saved every 5 epochs to `/kaggle/working/checkpoints/`
   - Best model saved as `best_model.pt`

## Troubleshooting

### "Checkpoint not found"
- Check the path in Cell 1
- List files: `!ls -lh /kaggle/input/`
- Update `CHECKPOINT_PATH` to correct location

### "Size mismatch" error
- Vocab size changed between training runs
- Make sure you're using the same `vocab.json`
- Check: `checkpoint['config']['model']['vocab_size']`

### Loss increases during fine-tuning
- Learning rate too high
- Try even smaller: `0.000005`
- Or increase dropout: `0.3` → `0.4`

### No improvement
- Model already converged
- Try more data or data augmentation
- Check if new data is significantly different

## Advanced: Custom Fine-tuning Settings

If you want more control, modify Cell 2:

```python
# Custom fine-tuning settings
if RESUME_TRAINING:
    learning_rate = 0.000005  # Even smaller LR
    num_epochs = 30           # Even fewer epochs
    dropout = 0.4             # Higher dropout to prevent overfitting
    
    # Update config
    config['model']['dropout'] = dropout
```

## Next Steps After Training

1. **Download best model:**
   ```python
   from IPython.display import FileLink
   FileLink('/kaggle/working/checkpoints/best_model.pt')
   ```

2. **Test the model:**
   ```python
   !python /kaggle/working/scripts/test_best_model.py \
       --checkpoint /kaggle/working/checkpoints/best_model.pt \
       --test_manifest /kaggle/working/konkani-10k/test_manifest.json
   ```

3. **Visualize training:**
   ```python
   !python /kaggle/working/scripts/generate_training_visualization.py \
       --log_dir /kaggle/working/logs
   ```
