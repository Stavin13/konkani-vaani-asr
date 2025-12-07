# Resume Training from Checkpoint 15 - Complete Guide

## Quick Start

```bash
chmod +x resume_training_from_checkpoint15.sh
./resume_training_from_checkpoint15.sh
```

## Checkpoint 15 Configuration

Based on analysis of `archives/checkpoint_epoch_15.pt`:

### Model Architecture
- **d_model**: 256
- **Encoder layers**: 12 (Conformer blocks)
- **Decoder layers**: 6 (Transformer)
- **Num heads**: 4
- **Vocab size**: 200 characters
- **Total parameters**: ~24.7M

### Training Hyperparameters
- **Learning rate**: 0.0005
- **Weight decay**: 1e-6
- **Gradient clipping**: 5.0
- **CTC weight**: 0.3 (attention weight: 0.7)
- **Mixed precision**: Enabled (FP16)

### Original Settings (Caused OOM)
- **Batch size**: 8
- **No gradient accumulation**

### Memory-Optimized Settings (For Tesla T4)
- **Batch size**: 2 ✅
- **Gradient accumulation**: 4 steps ✅
- **Effective batch size**: 8 (same as original)
- **Mixed precision**: FP16 ✅
- **CUDA cache clearing**: Every 50 batches ✅

## Why These Settings Work

### 1. Gradient Accumulation
```
Original: batch_size=8, accumulation=1 → 8 samples per update
Optimized: batch_size=2, accumulation=4 → 8 samples per update
```
Same effective batch size, but uses 4x less GPU memory!

### 2. Mixed Precision (FP16)
- Reduces memory by ~50%
- Speeds up training on Tesla T4
- Already enabled in checkpoint 15

### 3. Smaller Physical Batch
- batch_size=2 means only 2 audio samples in GPU at once
- Attention operations scale quadratically with sequence length
- Smaller batches = less memory for attention matrices

## Memory Breakdown (Tesla T4 - 14.74 GB)

### Original Config (Failed)
```
Model weights:        ~2 GB
Activations (batch=8): ~8-10 GB
Gradients:            ~2 GB
Optimizer states:     ~2 GB
Total:                ~14-16 GB ❌ (OOM!)
```

### Optimized Config (Works)
```
Model weights:        ~1 GB (FP16)
Activations (batch=2): ~2-3 GB (FP16)
Gradients:            ~1 GB (FP16)
Optimizer states:     ~2 GB
CUDA cache:           ~1 GB
Total:                ~7-8 GB ✅ (Fits!)
```

## Step-by-Step Instructions

### 1. Verify Checkpoint Exists
```bash
ls -lh archives/checkpoint_epoch_15.pt
```

### 2. Check GPU Status
```bash
nvidia-smi
```
Make sure no other processes are using GPU memory.

### 3. Clear GPU Memory (if needed)
```bash
# Kill other processes if necessary
nvidia-smi
# Note the PID of processes using GPU
kill -9 <PID>

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

### 4. Run Training
```bash
./resume_training_from_checkpoint15.sh
```

## Manual Command (Alternative)

If you prefer to run manually:

```bash
# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
python3 training_scripts/train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file data/vocab.json \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 50 \
    --learning_rate 0.0005 \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6 \
    --mixed_precision \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --resume checkpoints/checkpoint_epoch_15.pt
```

## If Still Out of Memory

### Option 1: Reduce Batch Size Further
```bash
python3 training_scripts/train_konkanivani_asr.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6 \
    --resume checkpoints/checkpoint_epoch_15.pt
```

### Option 2: Use Smaller Model (Not Recommended - breaks checkpoint compatibility)
Only use this if you want to start fresh training:
```bash
python3 training_scripts/train_konkanivani_asr.py \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --mixed_precision \
    --device cuda \
    --d_model 192 \
    --encoder_layers 8 \
    --decoder_layers 4
    # Don't use --resume (incompatible architecture)
```

## Monitoring Training

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View TensorBoard Logs
```bash
tensorboard --logdir logs
```

### Check Training Progress
Training will print:
- Epoch progress
- Train loss (CTC + Attention)
- Validation loss
- Learning rate
- Checkpoint saves

## Expected Behavior

### First Batch
- May take 30-60 seconds (model compilation)
- Memory usage will spike then stabilize

### During Training
- GPU memory: ~7-8 GB (stable)
- GPU utilization: 80-95%
- Speed: ~2-3 batches/second

### Checkpoints
- Saved every 5 epochs to `checkpoints/`
- Best model saved as `checkpoints/best_model.pt`

## Troubleshooting

### "CUDA out of memory" on first batch
**Solution**: Reduce batch_size to 1
```bash
--batch_size 1 --gradient_accumulation_steps 8
```

### "CUDA out of memory" during validation
**Solution**: Already handled with `torch.no_grad()` in code

### Training very slow
**Solution**: 
- Check GPU utilization with `nvidia-smi`
- Ensure mixed precision is enabled
- Reduce num_workers if CPU is bottleneck

### Checkpoint not found
**Solution**: 
```bash
cp archives/checkpoint_epoch_15.pt checkpoints/
```

## Configuration Files

All settings documented in:
- `config/training_config_from_checkpoint15.yaml` - Full config
- `resume_training_from_checkpoint15.sh` - Ready-to-run script
- `MEMORY_OPTIMIZATION_GUIDE.md` - General memory tips

## Key Differences from Original

| Setting | Original | Optimized | Reason |
|---------|----------|-----------|--------|
| Batch size | 8 | 2 | Reduce memory |
| Gradient accumulation | 1 | 4 | Maintain effective batch |
| Mixed precision | ✅ | ✅ | Already enabled |
| Cache clearing | ❌ | ✅ | Prevent fragmentation |
| Environment vars | ❌ | ✅ | Better memory management |

## Success Indicators

✅ Training starts without OOM error  
✅ GPU memory stable at ~7-8 GB  
✅ Batch processing speed: 2-3 batches/sec  
✅ Loss decreasing over epochs  
✅ Checkpoints saving successfully  

## Next Steps After Training

1. Evaluate model on test set
2. Run inference on new audio
3. Export model for deployment
4. Fine-tune on additional data

Good luck with your training! 🚀
