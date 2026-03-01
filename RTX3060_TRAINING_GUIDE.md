# RTX 3060 Konkani ASR Fine-tuning Guide

This guide provides optimized scripts for fine-tuning your Konkani ASR model on RTX 3060 6GB VRAM.

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
python quick_start_rtx3060.py
```
This script will:
- Check your system requirements
- Find your model and data files automatically
- Start training with optimal settings

### Option 2: Manual Setup
```bash
# 1. Check system and prepare
python setup_rtx3060_training.py

# 2. Start training
python finetune_rtx3060.py
```

## 📋 Requirements

### Hardware
- RTX 3060 (6GB VRAM) or similar
- 16GB+ system RAM recommended
- SSD storage for faster data loading

### Software
```bash
pip install -r requirements_rtx3060.txt
```

## 🎯 RTX 3060 Optimizations

### Memory Optimizations
- **Batch size**: 2 (with gradient accumulation = effective batch size 16)
- **Mixed precision**: Enabled (saves ~40% VRAM)
- **Gradient checkpointing**: Enabled
- **Audio length limit**: 8 seconds max
- **Mel spectrogram**: Optimized dimensions

### Performance Features
- **Gradient accumulation**: 8 steps
- **Dynamic padding**: Reduces memory waste
- **Memory clearing**: Periodic cleanup
- **Optimized data loading**: Reduced workers, no pin memory

## 📁 Expected File Structure

```
your_project/
├── best_model (1).pt          # Your trained model
├── data/
│   └── konkani-10k/           # Primary dataset
│       ├── train_manifest.json
│       ├── val_manifest.json
│       └── vocab.json
├── finetune_rtx3060.py        # Main training script
├── setup_rtx3060_training.py  # Setup checker
└── quick_start_rtx3060.py     # Auto-start script
```

## ⚙️ Training Parameters

### Default Settings (RTX 3060 Optimized)
```python
batch_size = 2                    # Small for 6GB VRAM
gradient_accumulation_steps = 8   # Effective batch size = 16
learning_rate = 0.00003          # Conservative for fine-tuning
epochs = 10                      # Quick fine-tuning
max_audio_length = 8 seconds     # Memory limit
mixed_precision = True           # Saves VRAM
```

### Custom Training
```bash
python finetune_rtx3060.py \
    --epochs 15 \
    --learning_rate 0.00005 \
    --checkpoint your_model.pt \
    --train_manifest your_train.json \
    --val_manifest your_val.json \
    --vocab_file your_vocab.json
```

## 📊 Expected Performance

### RTX 3060 Estimates
- **Training time**: ~2-3 hours for 10 epochs (depends on dataset size)
- **Memory usage**: ~5.5GB VRAM peak
- **Batch processing**: ~1-2 batches/second

### Memory Usage Breakdown
- Model: ~1.5GB
- Batch data: ~2GB
- Gradients: ~1.5GB
- Optimizer states: ~1GB
- Buffer: ~0.5GB

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors
```bash
# Reduce batch size further
python finetune_rtx3060.py --batch_size 1

# Or increase gradient accumulation
# Edit MEMORY_OPTIMIZED_SETTINGS in finetune_rtx3060.py:
gradient_accumulation_steps = 16  # Increase from 8
```

### Slow Training
```bash
# Check if using GPU
nvidia-smi

# Reduce audio length limit
# Edit in finetune_rtx3060.py:
max_audio_length = 16000 * 6  # 6 seconds instead of 8
```

### Data Loading Issues
```bash
# Check data paths
python setup_rtx3060_training.py

# Use alternative dataset
python finetune_rtx3060.py --train_manifest data/konkani-full/train.json
```

## 📈 Monitoring Training

### GPU Usage
```bash
# Monitor VRAM usage
nvidia-smi -l 1

# Or use built-in monitoring (shows in progress bar)
# Memory usage displayed as: mem: 5.2GB
```

### Training Progress
- Loss should decrease gradually
- Validation loss should follow training loss
- Best model saved automatically when validation improves

### Expected Loss Values
- Initial: ~8-12 (depends on model)
- After 5 epochs: ~3-6
- After 10 epochs: ~2-4
- Good convergence: <2.0

## 🎯 Advanced Options

### Model Architecture Tweaks
Edit `finetune_rtx3060.py` to modify model size:
```python
# Smaller model for more memory
'd_model': 96,           # Reduce from 128
'encoder_layers': 4,     # Reduce from 6
'decoder_layers': 3,     # Reduce from 4

# Larger model if you have headroom
'd_model': 160,          # Increase from 128
'encoder_layers': 8,     # Increase from 6
```

### Learning Rate Scheduling
```python
# Current: OneCycleLR (recommended)
# Alternative: ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
```

## 📝 Output Files

After training completes:
```
rtx3060_finetuned/
├── best_model_rtx3060.pt      # Best model checkpoint
├── training_log.txt           # Training progress
└── config.json               # Training configuration
```

## 🔍 Validation

Test your fine-tuned model:
```python
import torch
from your_model_code import load_model

# Load fine-tuned model
model = load_model('rtx3060_finetuned/best_model_rtx3060.pt')

# Test on sample audio
# (Add your inference code here)
```

## 💡 Tips for Best Results

1. **Data Quality**: Ensure audio files are 16kHz, mono
2. **Text Normalization**: Clean and consistent text formatting
3. **Balanced Dataset**: Mix of different speakers/conditions
4. **Validation**: Monitor validation loss to avoid overfitting
5. **Patience**: Let training complete - early stopping may miss improvements

## 🆘 Support

If you encounter issues:
1. Run `python setup_rtx3060_training.py` to check system
2. Check GPU memory with `nvidia-smi`
3. Verify data files exist and are readable
4. Try reducing batch size or audio length limits

## 📚 Additional Resources

- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [ASR Training Best Practices](https://pytorch.org/audio/stable/tutorials/asr_training_tutorial.html)

---

**Happy Training! 🎉**

Your RTX 3060 is ready to fine-tune Konkani ASR models efficiently!