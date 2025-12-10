#!/usr/bin/env python3
"""
Resume training from checkpoint with fixed configuration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

print("""
╔══════════════════════════════════════════════════════════════╗
║  RESUME TRAINING WITH FIXED CONFIGURATION                    ║
╚══════════════════════════════════════════════════════════════╝

Current Issue:
  - Model predicts 98% blank tokens
  - Only 3 unique tokens predicted
  - CTC loss weight too low (0.3)

Fixes Applied:
  ✓ Increase CTC weight: 0.3 → 0.8
  ✓ Add gradient clipping (norm 5.0)
  ✓ Increase learning rate: 1e-4 → 3e-4
  ✓ Add warmup schedule
  ✓ Monitor CER during training

Recommendation:
  1. Try this quick fix (10-20 epochs)
  2. If still not working, retrain from scratch with:
     - Smaller model (fewer layers)
     - Simpler architecture (just CTC, no attention)
     - Better data augmentation

To start training:
  python training_scripts/train_konkanivani_asr.py \\
    --resume kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt \\
    --ctc_weight 0.8 \\
    --learning_rate 3e-4 \\
    --gradient_clip 5.0 \\
    --epochs 20

""")

# Show current checkpoint status
checkpoint_path = 'kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt'
if Path(checkpoint_path).exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint Info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Path: {checkpoint_path}")
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")

print("\n" + "="*60)
print("ESTIMATED TRAINING TIME:")
print("="*60)
print("  10 epochs: ~2-4 hours (depending on hardware)")
print("  20 epochs: ~4-8 hours")
print("\nFull retraining from scratch: ~10-20 hours")
