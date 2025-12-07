# Fast Kaggle Upload - 3 Methods

## Your Situation
- Full project: ~14 GB
- Current upload: Taking forever
- Files already in Google Drive

## Method 1: Upload Minimal Package (FASTEST - 270MB)

### What to Upload:
- Code files only (training_scripts, models)
- Checkpoint (294 MB)
- Manifests and vocab
- **Skip audio files initially**

### Steps:

```bash
# Run the fixed script
chmod +x create_kaggle_minimal_fixed.sh
./create_kaggle_minimal_fixed.sh
```

This creates `kaggle_konkani_minimal.zip` (~270-500 MB depending on audio)

**Upload time**: 5-10 minutes (vs 1-2 hours for 14GB)

### Then in Kaggle Notebook:

```python
# Download audio files directly in Kaggle
# They're already public or you can use Drive API
```

---

## Method 2: Use Kaggle CLI (FASTER)

If you have fast internet, use Kaggle's command-line tool:

### Setup (one-time):

```bash
# Install Kaggle CLI
pip install kaggle

# Get API token
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/
```

### Upload:

```bash
# Create dataset metadata
cat > dataset-metadata.json << EOF
{
  "title": "konkani-asr-training",
  "id": "yourusername/konkani-asr-training",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Upload (much faster than web interface)
kaggle datasets create -p kaggle_minimal/
```

**Upload time**: 10-20 minutes (faster than web)

---

## Method 3: Mount Google Drive in Kaggle (SMARTEST)

**Don't upload at all!** Access files directly from Drive in Kaggle.

### Steps:

1. **Make Drive folder public** (or use service account)
2. **In Kaggle notebook**, download from Drive:

```python
# Install gdown
!pip install gdown

# Download from your public Drive link
!gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID

# Or use Drive API with authentication
```

**Upload time**: 0 minutes! Downloads in Kaggle (~10 min)

---

## Method 4: Split Upload (PRACTICAL)

Upload in stages:

### Stage 1: Code + Checkpoint (300 MB)
```bash
zip -r kaggle_stage1.zip \
  training_scripts/ \
  models/ \
  data/vocab.json \
  data/konkani-asr-v0/splits/manifests/ \
  archives/checkpoint_epoch_15.pt
```

**Upload this first** (5-10 min)

### Stage 2: Audio Files (later)
Upload audio separately or download in Kaggle

---

## Recommended Approach

### For You Right Now:

**Use Method 1 (Minimal Package)**

1. **Stop current upload** if still running
2. **Run the fixed script**:
   ```bash
   chmod +x create_kaggle_minimal_fixed.sh
   ./create_kaggle_minimal_fixed.sh
   ```
3. **Upload** `kaggle_konkani_minimal.zip` (270-500 MB)
4. **In Kaggle**, download audio from Drive if needed

### Why This Works:

- ✅ **270 MB vs 14 GB** - 50x smaller!
- ✅ **5-10 min upload** vs 1-2 hours
- ✅ **All essential files** included
- ✅ **Audio can be downloaded** in Kaggle from Drive

---

## What's in the Minimal Package:

```
kaggle_minimal/
├── training_scripts/          # Training code
├── models/                    # Model architecture
├── data/
│   ├── vocab.json            # Vocabulary
│   ├── audio_processing/     # Data loading code
│   └── konkani-asr-v0/
│       ├── audio/            # Audio files (if found)
│       └── splits/
│           └── manifests/    # Train/val splits
└── archives/
    └── checkpoint_epoch_15.pt # Checkpoint (294 MB)
```

**Total**: ~270-500 MB (depending on audio)

---

## If Audio Files Are Missing:

### Option A: Download in Kaggle from Drive

```python
# In Kaggle notebook
!pip install gdown

# Download from your public Drive folder
!gdown --folder https://drive.google.com/drive/folders/YOUR_AUDIO_FOLDER_ID
```

### Option B: Use Kaggle's Internet

```python
# If audio is hosted somewhere
!wget https://your-audio-source.com/audio.zip
!unzip audio.zip
```

### Option C: Upload Audio Separately

Create a second dataset just for audio files, then add both datasets to your notebook.

---

## Comparison:

| Method | Upload Time | Setup Complexity | Best For |
|--------|-------------|------------------|----------|
| **Minimal Package** | 5-10 min | Easy | ✅ You right now |
| Kaggle CLI | 10-20 min | Medium | Fast internet |
| Drive Mount | 0 min | Medium | Public files |
| Full Upload | 1-2 hours | Easy | Slow but simple |

---

## Quick Start (Do This Now):

```bash
# 1. Stop current upload (if running)

# 2. Create minimal package
chmod +x create_kaggle_minimal_fixed.sh
./create_kaggle_minimal_fixed.sh

# 3. Check size
ls -lh kaggle_konkani_minimal.zip

# 4. Upload to Kaggle
# Go to: https://www.kaggle.com/datasets
# Click: New Dataset
# Upload: kaggle_konkani_minimal.zip
# Wait: 5-10 minutes
# Done!
```

---

## After Upload:

1. **Create Kaggle notebook**
2. **Add your dataset**
3. **Upload** `KAGGLE_TRAINING.ipynb`
4. **Enable P100 GPU**
5. **Run training** (6-8 hours on P100)

---

## Troubleshooting:

### "Audio files not found"
- Check: `data/konkani-asr-v0/audio/`
- Or: `data/audio/`
- Or: `KonkaniRawSpeechCorpus/Data/`
- Script will auto-detect

### "Upload still slow"
- Check internet speed
- Try Kaggle CLI instead
- Or use Drive mount method

### "Package too large"
- Remove audio files from zip
- Download them in Kaggle instead
- Should be ~300 MB without audio

---

## Bottom Line:

**Stop the 14GB upload. Use the minimal package instead. 50x faster!**

```bash
./create_kaggle_minimal_fixed.sh
# Upload the 270MB file
# Start training in 10 minutes instead of 2 hours
```

🚀
