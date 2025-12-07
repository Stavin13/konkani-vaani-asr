# KonkaniVani ASR - Memory Optimization Guide

## Problem
Training fails with `CUDA out of memory` error on Tesla T4 (14.74 GiB GPU).

## Solutions (Try in Order)

### 1. **Immediate Fix - Run Optimized Script**
```bash
chmod +x run_training_memory_optimized.sh
./run_training_memory_optimized.sh
```

This script uses:
- Batch size: 2 (down from 8)
- Gradient accumulation: 4 steps (simulates batch size 8)
- Mixed precision (FP16) - saves ~50% memory
- Smaller model: 192 dims, 8 encoder, 4 decoder layers
- Periodic CUDA cache clearing

### 2. **If Still Out of Memory - Further Reduce Batch Size**
```bash
python3 training_scripts/train_konkanivani_asr.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision \
    --device cuda \
    --d_model 192 \
    --encoder_layers 8 \
    --decoder_layers 4 \
    --resume checkpoints/checkpoint_epoch_15.pt
```

### 3. **If Still Out of Memory - Use Even Smaller Model**
```bash
python3 training_scripts/train_konkanivani_asr.py \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --mixed_precision \
    --device cuda \
    --d_model 128 \
    --encoder_layers 6 \
    --decoder_layers 3 \
    --resume checkpoints/checkpoint_epoch_15.pt
```

### 4. **Clear GPU Memory Before Training**
```python
# Run this in Python before training
import torch
torch.cuda.empty_cache()
```

Or from command line:
```bash
python3 -c "import torch; torch.cuda.empty_cache()"
```

### 5. **Check What's Using GPU Memory**
```bash
nvidia-smi
```

If other processes are using GPU, kill them:
```bash
# Find process ID
nvidia-smi

# Kill process (replace PID with actual process ID)
kill -9 PID
```

## Memory Optimization Techniques Implemented

### ✅ Gradient Accumulation
- Accumulates gradients over multiple small batches
- Simulates larger batch size without memory overhead
- Example: batch_size=2 + accumulation=4 = effective batch_size=8

### ✅ Mixed Precision (FP16)
- Uses 16-bit floats instead of 32-bit
- Reduces memory by ~50%
- Speeds up training on modern GPUs
- Enable with `--mixed_precision` flag

### ✅ Smaller Model Architecture
- Reduced d_model from 256 → 192
- Reduced encoder layers from 12 → 8
- Reduced decoder layers from 6 → 4
- Still powerful enough for Konkani ASR

### ✅ Periodic Cache Clearing
- Clears CUDA cache every 50 batches
- Prevents memory fragmentation
- Automatically done in training loop

### ✅ Environment Variables
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
This helps PyTorch manage memory more efficiently.

## Model Size Comparison

| Configuration | Parameters | Memory (approx) |
|--------------|------------|-----------------|
| Original (d=256, 12+6 layers) | ~24.7M | ~8-10 GB |
| Optimized (d=192, 8+4 layers) | ~10-12M | ~4-6 GB |
| Minimal (d=128, 6+3 layers) | ~4-6M | ~2-3 GB |

## Troubleshooting

### Error: "Process 75955 has 14.17 GiB memory in use"
**Solution:** Another process is using GPU. Kill it or restart the runtime.

### Error: "Unable to register cuFFT/cuDNN/cuBLAS factory"
**Solution:** These are warnings, not errors. Safe to ignore.

### Training is very slow
**Solution:** 
- Ensure `--mixed_precision` is enabled
- Check `nvidia-smi` to verify GPU utilization is high
- Reduce `num_workers` in dataloader if CPU is bottleneck

### Validation also runs out of memory
**Solution:** Add this to validation loop (already implemented):
```python
with torch.no_grad():  # Disables gradient computation
    # validation code
```

## Best Practices

1. **Always use mixed precision on modern GPUs** (Tesla T4 supports it)
2. **Start with small batch size** and increase gradually
3. **Use gradient accumulation** to maintain effective batch size
4. **Monitor GPU memory** with `nvidia-smi` during training
5. **Save checkpoints frequently** in case of OOM crashes
6. **Clear cache** between experiments

## Quick Commands

```bash
# Check GPU status
nvidia-smi

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"

# Run optimized training
./run_training_memory_optimized.sh

# Resume from checkpoint with minimal memory
python3 training_scripts/train_konkanivani_asr.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision \
    --device cuda \
    --resume checkpoints/checkpoint_epoch_15.pt
```
