# Google Drive Setup Instructions

## Your Drive Folder
https://drive.google.com/drive/folders/1-chxczmcNooqLDtsFgQ8ZT8NvzFuFARr

## Quick Start

### 1. Upload Notebook to Colab
- Download `train_from_drive.ipynb` from your local project
- Go to [Google Colab](https://colab.research.google.com/)
- File → Upload notebook → Select `train_from_drive.ipynb`

### 2. Set GPU Runtime
```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### 3. Run Cells in Order
1. **Cell 1**: Check GPU
2. **Cell 2**: Install dependencies
3. **Cell 3**: Mount Google Drive
4. **Cell 4**: List Drive contents to find your project
5. **Cell 5**: Update `DRIVE_PROJECT_PATH` and copy project
6. **Cell 6**: Verify all files exist
7. **Cell 7-10**: Setup checkpoint and environment
8. **Cell 11**: 🚀 START TRAINING
9. **Cell 12-14**: Monitor and backup

## Important: Update Project Path

In **Cell 5**, you need to update this line:

```python
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/konkani"  # ⚠️ CHANGE THIS
```

### How to Find Your Path

After running **Cell 4**, you'll see a list of folders in your Drive. Look for your project folder name and update the path accordingly.

**Examples:**
```python
# If your project is directly in MyDrive
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/konkani"

# If it's in a subfolder
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/Projects/konkani"

# If it's in a shared drive
DRIVE_PROJECT_PATH = "/content/drive/Shareddrives/TeamDrive/konkani"
```

## Required Files in Your Drive Folder

Make sure these files are in your Drive folder:

```
konkani/
├── training_scripts/
│   └── train_konkanivani_asr.py
├── models/
│   └── konkanivani_asr.py
├── data/
│   ├── vocab.json
│   ├── audio_processing/
│   │   ├── dataset.py
│   │   └── text_tokenizer.py
│   └── konkani-asr-v0/
│       └── splits/
│           └── manifests/
│               ├── train.json
│               └── val.json
└── archives/
    └── checkpoint_epoch_15.pt
```

## Upload Options

### Option A: Upload Entire Project Folder
1. Compress your project: `zip -r konkani_project.zip konkani/`
2. Upload to Google Drive
3. In Colab, extract: `!unzip /content/drive/MyDrive/konkani_project.zip`

### Option B: Upload Individual Files
1. Create folder structure in Drive
2. Upload files to respective folders
3. Update `DRIVE_PROJECT_PATH` in notebook

### Option C: Use Shared Drive Link
1. Make sure the shared folder contains all required files
2. Access via: `/content/drive/Shareddrives/[DriveName]/[FolderPath]`

## Training Configuration

The notebook is pre-configured with:

| Setting | Value | Reason |
|---------|-------|--------|
| Batch size | 2 | Fits in 14GB GPU |
| Gradient accumulation | 4 | Effective batch = 8 |
| Mixed precision | FP16 | Saves ~50% memory |
| d_model | 256 | From checkpoint 15 |
| Encoder layers | 12 | From checkpoint 15 |
| Decoder layers | 6 | From checkpoint 15 |
| Resume from | Epoch 15 | Continue training |

## Expected Training Time

- **Per epoch**: ~15-20 minutes
- **Total (epochs 16-50)**: ~8-12 hours
- **GPU**: Tesla T4 (14GB)

## Backup Strategy

### Automatic Backups
Checkpoints are saved every 5 epochs to `/content/konkani/checkpoints/`

### Manual Backup to Drive
Run **Cell 14** periodically to backup to Drive:
```python
BACKUP_PATH = "/content/drive/MyDrive/konkanivani_backup"
```

This copies:
- All checkpoints
- Training logs
- TensorBoard data

## Monitoring Training

### GPU Usage (Cell 12)
```bash
!nvidia-smi
```
Expected:
- GPU Utilization: 80-95%
- Memory Used: ~7-8 GB / 14 GB
- Temperature: 60-80°C

### TensorBoard (Cell 13)
View training metrics:
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

### 1. "Files not found" Error

**Check Cell 4 output** to see your Drive structure, then update `DRIVE_PROJECT_PATH` in Cell 5.

```python
# Example: If you see "konkani-project" folder
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/konkani-project"
```

### 2. Out of Memory Error

**Run Cell 15 instead of Cell 11** - uses batch_size=1:
```python
--batch_size 1 \
--gradient_accumulation_steps 8 \
```

### 3. Drive Not Mounting

```python
# Unmount and remount
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

### 4. Slow File Copy

Copying from Drive to Colab can take 5-10 minutes for large projects. This is normal and only happens once.

### 5. Session Timeout

Colab free tier disconnects after:
- 12 hours of use
- 90 minutes of inactivity

**Solution**: 
- Keep tab open
- Backup every 5 epochs (automatic)
- Resume from latest checkpoint if disconnected

### 6. Permission Denied

Make sure you have access to the shared Drive folder. If it's a shared link, you may need to:
1. Open the link in browser
2. Add to "My Drive" (right-click → Add shortcut to Drive)
3. Access via `/content/drive/MyDrive/[ShortcutName]`

## After Training

### 1. Download Best Model
Run **Cell 16** to download `best_model.pt`

### 2. Backup Everything
Run **Cell 14** to backup to Drive

### 3. Copy Back to Original Drive Folder
```python
!cp -r checkpoints /content/drive/MyDrive/konkani/
!cp -r logs /content/drive/MyDrive/konkani/
```

## Tips for Success

1. ✅ **Keep Colab tab open** - Prevents disconnection
2. ✅ **Backup every 5 epochs** - Automatic in training script
3. ✅ **Monitor GPU usage** - Should be 80-95%
4. ✅ **Use TensorBoard** - Visualize training progress
5. ✅ **Save notebook** - File → Save a copy in Drive

## Cost Options

| Option | Cost | GPU Time | Best For |
|--------|------|----------|----------|
| Colab Free | $0 | 12h max | This training (8-12h) |
| Colab Pro | $10/mo | 24h max | Longer experiments |
| Colab Pro+ | $50/mo | Background | Multiple runs |

**Recommendation**: Colab Free should work for this training if you keep the tab open.

## Quick Reference

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

# Resume from different checkpoint
# Change --resume to:
--resume checkpoints/checkpoint_epoch_20.pt
```

## Support Checklist

Before asking for help, verify:
- [ ] GPU is allocated (Cell 1 shows Tesla T4)
- [ ] Drive is mounted (Cell 3 succeeds)
- [ ] All files exist (Cell 6 shows all ✅)
- [ ] Checkpoint is valid (Cell 8 shows config)
- [ ] Path is correct (Cell 5 copies successfully)

## Next Steps

1. Upload `train_from_drive.ipynb` to Colab
2. Set GPU runtime
3. Run cells 1-10 to setup
4. Run cell 11 to start training
5. Monitor with cells 12-13
6. Backup with cell 14

Good luck! 🚀
