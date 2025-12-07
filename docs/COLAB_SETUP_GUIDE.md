# Google Colab Training Setup Guide

## Quick Start

1. **Upload to Colab**: Upload `train_konkanivani_colab.ipynb` to Google Colab
2. **Set GPU**: Runtime → Change runtime type → GPU (T4)
3. **Run cells sequentially** from top to bottom

## Pre-requisites

### Files to Upload/Have Ready

1. **Project files** (one of these methods):
   - Upload `konkani_project.zip` to Google Drive
   - Clone from GitHub repository
   - Upload files directly to Colab

2. **Required files in project**:
   ```
   training_scripts/train_konkanivani_asr.py
   models/konkanivani_asr.py
   data/audio_processing/dataset.py
   data/audio_processing/text_tokenizer.py
   data/vocab.json
   data/konkani-asr-v0/splits/manifests/train.json
   data/konkani-asr-v0/splits/manifests/val.json
   archives/checkpoint_epoch_15.pt
   ```

## Step-by-Step Instructions

### 1. Open Colab Notebook
- Go to [Google Colab](https://colab.research.google.com/)
- Upload `train_konkanivani_colab.ipynb`
- Or: File → Upload notebook

### 2. Set GPU Runtime
```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### 3. Run Setup Cells (Cells 1-8)
These cells will:
- Check GPU availability
- Install dependencies
- Mount Google Drive
- Upload/clone project
- Verify files
- Copy checkpoint
- Set environment variables
- Clear GPU memory

### 4. Start Training (Cell 9)
The training will:
- Resume from epoch 16
- Use batch_size=2 with gradient accumulation
- Save checkpoints every 5 epochs
- Run for epochs 16-50

### 5. Monitor Progress
- **Cell 10**: Check GPU usage with `nvidia-smi`
- **Cell 11**: View TensorBoard logs
- **Cell 12**: Backup to Google Drive

## Configuration Details

### Memory-Optimized Settings
```python
--batch_size 2                      # Reduced from 8
--gradient_accumulation_steps 4     # Effective batch = 8
--mixed_precision                   # FP16 training
--d_model 256                       # From checkpoint
--encoder_layers 12                 # From checkpoint
--decoder_layers 6                  # From checkpoint
```

### Expected Memory Usage (Tesla T4)
- Total GPU: 14.74 GB
- Model + Training: ~7-8 GB
- Available: ~6-7 GB buffer

## Upload Methods

### Method 1: Upload Zip to Drive (Recommended)
```python
# In Colab cell:
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/konkani_project.zip .
!unzip -q konkani_project.zip
%cd konkani
```

### Method 2: Clone from GitHub
```python
!git clone https://github.com/yourusername/konkani-asr.git
%cd konkani-asr
```

### Method 3: Direct Upload
```python
from google.colab import files
uploaded = files.upload()  # Upload zip file
!unzip -q konkani_project.zip
```

## Backup Strategy

### Automatic Backup (Every 5 Epochs)
The training script saves checkpoints to `checkpoints/` every 5 epochs.

### Manual Backup to Drive
Run Cell 12 periodically:
```python
!cp -r checkpoints /content/drive/MyDrive/konkanivani_backup/
!cp -r logs /content/drive/MyDrive/konkanivani_backup/
```

### Download Checkpoints
```python
from google.colab import files
files.download('checkpoints/best_model.pt')
```

## Monitoring Training

### GPU Usage
```bash
!nvidia-smi
```
Expected:
- GPU Utilization: 80-95%
- Memory Used: ~7-8 GB
- Temperature: 60-80°C

### TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir logs
```
View:
- Training loss
- Validation loss
- Learning rate
- CTC vs Attention loss

### Training Output
```
Epoch 16/50
  Train Loss: 2.3456 (CTC: 1.2345)
  Val Loss: 2.4567 (CTC: 1.3456)
✅ Saved best model with val_loss: 2.4567
```

## Troubleshooting

### 1. Out of Memory Error

**Solution A**: Reduce batch size
```python
!python3 training_scripts/train_konkanivani_asr.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision \
    # ... other args
```

**Solution B**: Clear cache
```python
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
```

**Solution C**: Restart runtime
```
Runtime → Restart runtime
```

### 2. Checkpoint Not Found

```python
# Check if file exists
!ls -lh archives/checkpoint_epoch_15.pt

# Copy to checkpoints directory
!mkdir -p checkpoints
!cp archives/checkpoint_epoch_15.pt checkpoints/
```

### 3. Module Not Found Error

```python
# Add project to Python path
import sys
sys.path.append('/content/konkani')  # Adjust path as needed
```

### 4. Session Timeout

Colab free tier disconnects after:
- 12 hours of continuous use
- 90 minutes of inactivity

**Solutions**:
- Backup to Drive every 5 epochs (automatic)
- Use Colab Pro ($10/month) for longer sessions
- Resume from latest checkpoint

### 5. Slow Training

**Check GPU is being used**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show Tesla T4
```

**Verify mixed precision**:
Check training output for "✅ Using mixed precision training (FP16)"

### 6. Data Loading Errors

```python
# Verify manifest files
!head -5 data/konkani-asr-v0/splits/manifests/train.json

# Check audio files exist
!ls data/konkani-asr-v0/audio/ | head -10
```

## Expected Training Time

### Tesla T4 GPU
- **Per epoch**: ~15-20 minutes (2549 training samples)
- **Total (epochs 16-50)**: ~8-12 hours
- **Batches per second**: ~2-3

### Progress Indicators
```
Epoch 16: 100% 319/319 [15:23<00:00, 2.89s/it]
Validation: 100% 40/40 [01:15<00:00, 1.88s/it]
```

## After Training

### 1. Download Best Model
```python
from google.colab import files
files.download('checkpoints/best_model.pt')
```

### 2. Backup Everything to Drive
```python
!cp -r checkpoints /content/drive/MyDrive/konkanivani_backup/
!cp -r logs /content/drive/MyDrive/konkanivani_backup/
```

### 3. Test Inference
Run Cell 14 to load model and test

### 4. Evaluate on Test Set
```python
!python3 scripts/evaluate_model.py \
    --checkpoint checkpoints/best_model.pt \
    --test_manifest data/konkani-asr-v0/splits/manifests/test.json
```

## Tips for Success

1. **Keep Colab tab open** - Prevents disconnection
2. **Backup frequently** - Every 5 epochs to Drive
3. **Monitor GPU usage** - Should be 80-95%
4. **Check logs** - Use TensorBoard for visualization
5. **Save notebook** - File → Save a copy in Drive
6. **Use Colab Pro** - For longer uninterrupted training

## Cost Comparison

| Option | Cost | GPU Time | Disconnects |
|--------|------|----------|-------------|
| Colab Free | $0 | 12h max | Yes (90min idle) |
| Colab Pro | $10/mo | 24h max | Less frequent |
| Colab Pro+ | $50/mo | Background execution | Rare |

For this training (8-12 hours), **Colab Free** should work if you:
- Keep tab open
- Backup every 5 epochs
- Resume if disconnected

## Quick Reference Commands

```python
# Check GPU
!nvidia-smi

# Clear memory
import torch; torch.cuda.empty_cache()

# Backup to Drive
!cp -r checkpoints /content/drive/MyDrive/konkanivani_backup/

# Download model
from google.colab import files
files.download('checkpoints/best_model.pt')

# View logs
%load_ext tensorboard
%tensorboard --logdir logs

# Resume training (if interrupted)
!python3 training_scripts/train_konkanivani_asr.py \
    --resume checkpoints/checkpoint_epoch_XX.pt \
    # ... other args
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all files are uploaded correctly
4. Check GPU is allocated and available
5. Ensure checkpoint matches model architecture

Good luck with your training! 🚀
