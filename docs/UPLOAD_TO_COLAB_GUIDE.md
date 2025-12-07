# How to Upload Project to Google Colab

## Quick Start - Use `train_colab_simple.ipynb`

This is the easiest method! Just upload your project and run.

## Step 1: Prepare Your Project

On your local machine, create a zip file:

```bash
# Navigate to your project directory
cd /path/to/konkani

# Create zip file (exclude unnecessary files)
zip -r konkani_project.zip . \
  -x "*.git*" \
  -x "*__pycache__*" \
  -x "*.pyc" \
  -x ".venv/*" \
  -x "*.DS_Store" \
  -x "outputs/*" \
  -x "logs/*"
```

Or use this simpler command:
```bash
zip -r konkani_project.zip \
  training_scripts/ \
  models/ \
  data/ \
  archives/checkpoint_epoch_15.pt
```

## Step 2: Upload to Google Colab

### Method A: Direct Upload (Fastest for first time)

1. Open `train_colab_simple.ipynb` in Google Colab
2. Set Runtime → GPU (T4)
3. Run Cell 2 (Option A) - it will prompt for file upload
4. Select your `konkani_project.zip`
5. Wait for upload and extraction
6. Continue with remaining cells

### Method B: Via Google Drive (Better for repeated use)

1. Upload `konkani_project.zip` to Google Drive
2. Open `train_colab_simple.ipynb` in Google Colab
3. Set Runtime → GPU (T4)
4. Run Cell 2 (Option B)
5. Update the path to your zip file location
6. Continue with remaining cells

## Step 3: Verify Files

After extraction, Cell 5 will check for all required files:

```
✅ training_scripts/train_konkanivani_asr.py
✅ models/konkanivani_asr.py
✅ data/audio_processing/dataset.py
✅ data/audio_processing/text_tokenizer.py
✅ data/vocab.json
✅ data/konkani-asr-v0/splits/manifests/train.json
✅ data/konkani-asr-v0/splits/manifests/val.json
✅ archives/checkpoint_epoch_15.pt
```

If any files are missing, check your zip file contents.

## Step 4: Start Training

Run Cell 9 to start training with optimized settings:
- Batch size: 2
- Gradient accumulation: 4
- Mixed precision: FP16
- Resume from epoch 15

## Required Files Checklist

Make sure your zip contains:

```
konkani_project/
├── training_scripts/
│   └── train_konkanivani_asr.py
├── models/
│   └── konkanivani_asr.py
├── data/
│   ├── vocab.json
│   ├── audio_processing/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── text_tokenizer.py
│   │   └── audio_processor.py
│   └── konkani-asr-v0/
│       ├── audio/
│       │   └── [your audio files]
│       └── splits/
│           └── manifests/
│               ├── train.json
│               └── val.json
└── archives/
    └── checkpoint_epoch_15.pt
```

## File Size Considerations

### If your project is large (>1GB):

**Option 1: Upload only essential files**
```bash
# Create minimal zip without audio files
zip -r konkani_minimal.zip \
  training_scripts/ \
  models/ \
  data/vocab.json \
  data/audio_processing/ \
  data/konkani-asr-v0/splits/ \
  archives/checkpoint_epoch_15.pt
```

Then upload audio files separately to Drive.

**Option 2: Use Google Drive**
- Upload full project to Drive once
- Use `train_from_drive.ipynb` instead
- Colab will copy from Drive (faster for repeated use)

## Troubleshooting

### "Files not found" after extraction

**Check extraction path:**
```python
!ls -la /content/
```

The zip might extract to:
- `/content/konkani/`
- `/content/konkani_project/`
- `/content/` (if zip was created from inside the folder)

**Update Cell 3** with the correct path.

### Upload fails or times out

**Solution**: Use Google Drive method instead
1. Upload zip to Drive
2. Use Cell 2 Option B
3. Copy from Drive (more reliable)

### Zip file too large

**Solution**: Create minimal zip without audio
```bash
# Exclude audio files
zip -r konkani_minimal.zip . -x "data/konkani-asr-v0/audio/*"
```

Then upload audio separately or use Drive.

### Wrong folder structure after extraction

**Check what's inside:**
```python
!unzip -l konkani_project.zip | head -20
```

Make sure paths start with the files directly, not nested folders.

## Alternative: Manual File Upload

If zip doesn't work, upload files manually:

```python
from google.colab import files

# Create structure
!mkdir -p training_scripts models data/audio_processing archives

# Upload files one by one
uploaded = files.upload()  # Select train_konkanivani_asr.py
!mv train_konkanivani_asr.py training_scripts/

# Repeat for other files...
```

## Best Practices

1. **Test locally first**: Make sure training works on your machine
2. **Verify checkpoint**: Ensure checkpoint_epoch_15.pt is included
3. **Check file paths**: All paths in manifests should be relative
4. **Exclude large files**: Don't include logs, outputs, .git
5. **Use Drive for large projects**: Upload once, use many times

## Quick Commands

```bash
# Create zip (local machine)
zip -r konkani_project.zip training_scripts/ models/ data/ archives/

# Check zip contents
unzip -l konkani_project.zip

# Get zip size
ls -lh konkani_project.zip

# Extract specific files only
unzip konkani_project.zip "training_scripts/*" "models/*"
```

## Expected Upload Times

| File Size | Direct Upload | Via Drive |
|-----------|---------------|-----------|
| < 100 MB  | 1-2 min      | 2-3 min   |
| 100-500 MB| 3-5 min      | 5-8 min   |
| 500 MB-1 GB| 5-10 min    | 10-15 min |
| > 1 GB    | 10-20 min    | 15-30 min |

## After Upload

Once files are uploaded and verified:
1. Cell 6: Prepare checkpoint
2. Cell 7: Verify checkpoint config
3. Cell 8: Setup environment
4. Cell 9: Start training (8-12 hours)
5. Cell 12: Backup periodically

## Summary

**Easiest method**: 
1. Create zip: `zip -r konkani_project.zip training_scripts/ models/ data/ archives/`
2. Upload `train_colab_simple.ipynb` to Colab
3. Run Cell 2 Option A and upload zip
4. Run remaining cells

**For repeated use**:
1. Upload zip to Google Drive once
2. Use `train_from_drive.ipynb`
3. Update Drive path
4. Faster for multiple training runs

Good luck! 🚀
