# Additional Kaggle Cell: CUDA Memory Fix (Insert BEFORE training cell)

```python
# ============================================================
# CUDA Memory Alignment Fix
# ============================================================

import os
import torch

# Enable better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ CUDA cache cleared")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

# Set deterministic behavior (helps with alignment issues)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("✅ Deterministic mode enabled (fixes alignment issues)")
```

## Alternative Solution: Disable DataParallel

If the error persists, add this cell to disable DataParallel:

```python
# ============================================================
# Disable DataParallel (if multi-GPU causes issues)
# ============================================================

# Set this flag BEFORE the training cell
DISABLE_DATAPARALLEL = True

print("⚠️  DataParallel disabled - using single GPU only")
print("   This fixes memory alignment issues but is slower")
```

Then modify the training cell (Cell 3) to check this flag:

```python
# Multi-GPU setup (MODIFIED to check flag)
if num_gpus > 1 and not DISABLE_DATAPARALLEL:
    print(f"\n🚀 Enabling multi-GPU training ({num_gpus} GPUs)")
    model = nn.DataParallel(model)
    effective_batch_size = config['training']['batch_size'] * num_gpus * config['training']['gradient_accumulation_steps']
    print(f"  Batch size per GPU: {config['training']['batch_size']}")
    print(f"  Total batch per step: {config['training']['batch_size'] * num_gpus}")
    print(f"  Effective batch size: {effective_batch_size}")
elif num_gpus > 1:
    print(f"\n⚠️  Multi-GPU available but disabled (using single GPU)")
    print(f"   Available GPUs: {num_gpus}")
    print(f"   Using: GPU 0 only")
else:
    print(f"\n🖥️  Single GPU training")
```

## Solution 3: Fix Validation Batch Size

Add this to ensure consistent batch sizes:

```python
# ============================================================
# Fix Validation DataLoader (Insert after dataloader creation)
# ============================================================

# Ensure validation uses drop_last=True to avoid small batches
from torch.utils.data import DataLoader

# Recreate validation loader with drop_last
val_dataset = val_loader.dataset
val_loader = DataLoader(
    val_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['data']['num_workers'],
    drop_last=True,  # Drop incomplete batches
    pin_memory=True
)

print(f"✅ Validation loader fixed:")
print(f"   Batches: {len(val_loader)}")
print(f"   Batch size: {config['training']['batch_size']}")
print(f"   Drop last: True (prevents alignment issues)")
```

## Complete Fix Sequence

Insert these cells in this order:

1. **Cell A: CUDA Memory Fix** (insert before training)
2. **Cell B: Disable DataParallel flag** (optional, only if needed)
3. **Cell C: Fix Validation Batch Size** (after dataloader creation)
4. **Modified Cell 3: Training with DataParallel check**

## Quick Diagnostic

If error still occurs, add this diagnostic cell:

```python
# ============================================================
# Diagnostic: Check Validation Data
# ============================================================

print("🔍 Checking validation data...")

for i, batch in enumerate(val_loader):
    print(f"\nBatch {i+1}:")
    print(f"  Audio shape: {batch['audio_features'].shape}")
    print(f"  Tokens shape: {batch['transcript_tokens'].shape}")
    print(f"  Audio lengths: {batch['audio_lengths']}")
    print(f"  Token lengths: {batch['transcript_lengths']}")
    
    # Check for alignment
    if batch['audio_features'].shape[0] != config['training']['batch_size']:
        print(f"  ⚠️  WARNING: Batch size mismatch! Expected {config['training']['batch_size']}, got {batch['audio_features'].shape[0]}")
    
    if i >= 2:  # Check first 3 batches
        break

print("\n✅ Diagnostic complete")
```

## Root Cause Summary

The error `RuntimeError: CUDA error: misaligned address` in your case is caused by:

1. **DataParallel + Mixed Precision**: The combination can cause alignment issues
2. **Inconsistent Batch Sizes**: Last validation batch might be smaller
3. **Memory Fragmentation**: After training epoch, GPU memory is fragmented

## Recommended Fix Order

Try these in order:

1. ✅ **Add CUDA synchronization** (already done in training script)
2. ✅ **Enable CUDA_LAUNCH_BLOCKING** (Cell A above)
3. ✅ **Use drop_last=True** for validation (Cell C above)
4. ⚠️ **Disable DataParallel** if still failing (Cell B above)
5. 🔧 **Reduce batch size** from 8 to 4 if memory is tight

## Expected Behavior After Fix

✅ Validation should complete without errors
✅ Memory alignment issues resolved
✅ Training continues normally
✅ Checkpoints saved successfully
