# Complete Setup Guide: Training KonkaniVani with Google One + Colab

This guide will help you train your ASR model using your Google One account (2TB storage) with Google Colab's free GPU.

---

## 📋 What You'll Need

- ✅ Google One account (2TB storage)
- ✅ Google Colab (free)
- ✅ Your Mac with the project files
- ✅ Stable internet connection
- ✅ ~2-3 hours for initial setup
- ✅ ~12 hours for training to complete

---

## PART 1: Upload Data to Google One Drive

### Option A: Using Google Drive Desktop App (RECOMMENDED - Fastest)

**Step 1.1: Install Google Drive for Desktop**

1. Go to: https://www.google.com/drive/download/
2. Download "Drive for Desktop" for Mac
3. Install and open the app
4. Sign in with your **Google One account**

**Step 1.2: Upload Your Project Files**

Once Drive is mounted, open Terminal and run:

```bash
# Navigate to your project
cd /Volumes/data\&proj/konkani

# Create a folder in your Google Drive
mkdir -p ~/Library/CloudStorage/GoogleDrive-YOUR_EMAIL@gmail.com/MyDrive/konkanivani_training

# Copy audio data (this will take 30-60 minutes for 16GB)
cp -r data/konkani-asr-v0 ~/Library/CloudStorage/GoogleDrive-YOUR_EMAIL@gmail.com/MyDrive/konkanivani_training/

# Copy checkpoint
cp checkpoint_epoch_15.pt ~/Library/CloudStorage/GoogleDrive-YOUR_EMAIL@gmail.com/MyDrive/konkanivani_training/

# Copy vocab
cp vocab.json ~/Library/CloudStorage/GoogleDrive-YOUR_EMAIL@gmail.com/MyDrive/konkanivani_training/
```

**Replace `YOUR_EMAIL@gmail.com` with your actual Google One email!**

The sync will happen automatically in the background. You can check progress in the Drive app menu bar icon.

---

### Option B: Using Web Browser (Slower but simpler)

**Step 1.1: Go to Google Drive**

1. Open browser: https://drive.google.com
2. Sign in with your **Google One account**
3. Create a new folder: `konkanivani_training`

**Step 1.2: Upload Files**

1. Click "New" → "Folder upload"
2. Select `data/konkani-asr-v0` folder (this will take 1-2 hours)
3. Wait for upload to complete
4. Upload `checkpoint_epoch_15.pt` file
5. Upload `vocab.json` file

**Verify uploads:**
- Check that `konkanivani_training/konkani-asr-v0/data/processed_segments_diarized/audio_segments/` has audio files
- Check that `konkanivani_training/konkani-asr-v0/splits/manifests/` has train.json, val.json, test.json
- Check that `checkpoint_epoch_15.pt` is there (294MB)
- Check that `vocab.json` is there

---

## PART 2: Prepare Code Package

On your Mac, create the lightweight code package:

```bash
cd /Volumes/data\&proj/konkani

# Run the script I created
./create_colab_package.sh

# This creates: konkani_code.zip (38MB)
```

You'll upload this to Colab in the next step.

---

## PART 3: Set Up Google Colab

**Step 3.1: Open Colab**

1. Go to: https://colab.research.google.com
2. Sign in with your **Google One account** (same one with the data)
3. Click "File" → "Upload notebook"
4. Upload `train_konkanivani_colab.ipynb` from your Mac

**Step 3.2: Enable GPU**

1. Click "Runtime" → "Change runtime type"
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (free tier)
4. Click "Save"

**Step 3.3: Verify GPU**

Run this cell:
```python
!nvidia-smi
```

You should see GPU info (Tesla T4, ~15GB memory).

---

## PART 4: Run Training in Colab

### Cell 1: Install Dependencies

```python
print("📦 Installing dependencies...")
!pip install -q torch torchaudio tensorboard jiwer pyyaml soundfile
print("✅ Dependencies installed!")
```

---

### Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

print("\n📂 Checking your uploaded files...\n")

# Check audio data
import os
audio_path = "/content/drive/MyDrive/konkanivani_training/konkani-asr-v0/data/processed_segments_diarized/audio_segments"
if os.path.exists(audio_path):
    audio_count = len([f for f in os.listdir(audio_path) if f.endswith('.wav')])
    print(f"✅ Audio files: {audio_count} files found")
else:
    print(f"❌ Audio files not found at: {audio_path}")
    print("   Please check the path in your Drive!")

# Check manifests
manifest_path = "/content/drive/MyDrive/konkanivani_training/konkani-asr-v0/splits/manifests"
if os.path.exists(manifest_path):
    print(f"✅ Manifests found")
    !ls -lh {manifest_path}/*.json
else:
    print(f"❌ Manifests not found at: {manifest_path}")

# Check checkpoint
checkpoint_path = "/content/drive/MyDrive/konkanivani_training/checkpoint_epoch_15.pt"
if os.path.exists(checkpoint_path):
    print(f"✅ Checkpoint found (294MB)")
else:
    print(f"❌ Checkpoint not found at: {checkpoint_path}")

# Check vocab
vocab_path = "/content/drive/MyDrive/konkanivani_training/vocab.json"
if os.path.exists(vocab_path):
    print(f"✅ Vocab found")
else:
    print(f"❌ Vocab not found at: {vocab_path}")
```

**⚠️ IMPORTANT:** If any files are missing, go back to Part 1 and verify your uploads!

---

### Cell 3: Upload Code Package

```python
from google.colab import files
import os

print("📤 Please upload konkani_code.zip (38MB)")
print("   This file is in your Mac's konkani folder\n")

uploaded = files.upload()

if 'konkani_code.zip' in uploaded:
    print("\n📂 Extracting code...")
    !unzip -q konkani_code.zip -d /content/
    print("✅ Code extracted!")
    
    print("\n📋 Checking extracted files:")
    !ls -la /content/
    !ls -la /content/models/ 2>/dev/null || echo "⚠️  models folder missing"
    !ls -la /content/data/audio_processing/ 2>/dev/null || echo "⚠️  data/audio_processing missing"
else:
    print("❌ Please upload konkani_code.zip")
```

---

### Cell 4: Link Data from Drive (Fast - No Copying!)

```python
import os

print("🔗 Creating symbolic links to your Drive data...\n")

# Create directory structure
!mkdir -p /content/data/konkani-asr-v0/data/processed_segments_diarized
!mkdir -p /content/data/konkani-asr-v0/splits
!mkdir -p /content/checkpoints
!mkdir -p /content/logs

# Link audio files (instant - no copying!)
drive_audio = "/content/drive/MyDrive/konkanivani_training/konkani-asr-v0/data/processed_segments_diarized/audio_segments"
local_audio = "/content/data/konkani-asr-v0/data/processed_segments_diarized/audio_segments"

if os.path.exists(drive_audio):
    !ln -s {drive_audio} {local_audio}
    audio_count = len([f for f in os.listdir(drive_audio) if f.endswith('.wav')])
    print(f"✅ Linked {audio_count} audio files from Drive")
else:
    print(f"❌ Audio path not found: {drive_audio}")

# Link manifests
drive_manifests = "/content/drive/MyDrive/konkanivani_training/konkani-asr-v0/splits/manifests"
local_manifests = "/content/data/konkani-asr-v0/splits/manifests"

if os.path.exists(drive_manifests):
    !ln -s {drive_manifests} {local_manifests}
    print("✅ Linked manifest files from Drive")
    !ls -lh {local_manifests}/*.json
else:
    print(f"❌ Manifest path not found: {drive_manifests}")

# Copy checkpoint (small file, so we copy it)
drive_checkpoint = "/content/drive/MyDrive/konkanivani_training/checkpoint_epoch_15.pt"
if os.path.exists(drive_checkpoint):
    !cp {drive_checkpoint} /content/checkpoints/
    print("\n✅ Copied checkpoint to /content/checkpoints/")
    !ls -lh /content/checkpoints/
else:
    print(f"❌ Checkpoint not found: {drive_checkpoint}")

# Copy vocab
drive_vocab = "/content/drive/MyDrive/konkanivani_training/vocab.json"
if os.path.exists(drive_vocab):
    !cp {drive_vocab} /content/vocab.json
    print("✅ Copied vocab.json")
else:
    print(f"❌ Vocab not found: {drive_vocab}")

print("\n✅ Setup complete! Ready to train.")
```

---

### Cell 5: Start Training (Resume from Epoch 15)

```python
import os
import re

print("🚀 Starting training...\n")

# Verify checkpoint
checkpoint_path = "/content/checkpoints/checkpoint_epoch_15.pt"
if os.path.exists(checkpoint_path):
    print(f"✅ Found checkpoint: checkpoint_epoch_15.pt")
    print(f"   Will resume from Epoch 16\n")
    resume_flag = f"--resume {checkpoint_path}"
else:
    print("⚠️  No checkpoint found, starting from scratch\n")
    resume_flag = ""

# Start training with GPU
!python3 /content/train_konkanivani_asr.py \
    --train_manifest /content/data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest /content/data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file /content/vocab.json \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 0.0005 \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6 \
    --checkpoint_dir /content/checkpoints \
    --log_dir /content/logs \
    {resume_flag}
```

**This will run for ~12 hours.** Keep the browser tab open!

---

### Cell 6: Monitor Training (Run in parallel)

Open a new cell and run this while training:

```python
# Check latest checkpoint
!ls -lth /content/checkpoints/ | head -5

# Check training logs
!tail -50 /content/logs/training.log 2>/dev/null || echo "Log file not created yet"
```

---

### Cell 7: Backup Checkpoints to Drive (Run every few hours)

```python
import os
from datetime import datetime

print("💾 Backing up checkpoints to Google Drive...\n")

# Create backup folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/content/drive/MyDrive/konkanivani_training/backups/backup_{timestamp}"

!mkdir -p {backup_path}

# Copy all checkpoints
!cp -r /content/checkpoints/* {backup_path}/
!cp -r /content/logs {backup_path}/

print(f"✅ Backed up to: {backup_path}")
!ls -lh {backup_path}/
```

**Run this cell every 2-3 hours to save your progress!**

---

### Cell 8: After Training - Download Best Model

```python
from google.colab import files
import os

print("📦 Preparing best model for download...\n")

# Create a package with the best model
!mkdir -p /content/final_model
!cp /content/checkpoints/best_model.pt /content/final_model/
!cp /content/vocab.json /content/final_model/
!cp -r /content/models /content/final_model/

# Zip it
!cd /content && zip -r final_konkanivani_model.zip final_model/

print("✅ Model packaged!\n")
print("File size:")
!ls -lh /content/final_konkanivani_model.zip

print("\n📥 Downloading...")
files.download('/content/final_konkanivani_model.zip')

print("\n✅ Download complete!")
print("   Also saved to: /content/drive/MyDrive/konkanivani_training/backups/")
```

---

## PART 5: Keep Colab Alive

Colab disconnects after ~90 minutes of inactivity. To prevent this:

**Option 1: Browser Extension**
- Install "Colab Auto Clicker" extension for Chrome/Firefox
- It will keep the session alive

**Option 2: JavaScript Console**

Press F12 in browser, go to Console tab, paste this:

```javascript
function KeepAlive() {
    console.log("Keeping Colab alive - " + new Date());
    document.querySelector("colab-connect-button")?.click();
}
setInterval(KeepAlive, 60000); // Every 60 seconds
```

**Option 3: Check periodically**
- Just click on the Colab tab every hour or so

---

## 📊 Expected Timeline

| Step | Time | What's Happening |
|------|------|------------------|
| Upload to Drive | 1-2 hours | Uploading 16GB audio data |
| Colab setup | 5 minutes | Installing dependencies, linking files |
| Training (Epoch 16-50) | ~12 hours | GPU training on Colab |
| **Total** | **~14 hours** | Most of it is automated |

---

## 🔧 Troubleshooting

### "Files not found in Drive"
- Check you're signed into the correct Google One account
- Verify folder structure: `MyDrive/konkanivani_training/konkani-asr-v0/`
- Make sure upload completed (check Drive web interface)

### "Out of memory" error
- Reduce batch size: `--batch_size 8` or `--batch_size 4`
- This will make training slower but use less memory

### "Runtime disconnected"
- Use the keep-alive methods above
- If it disconnects, just reconnect and resume from latest checkpoint
- Your checkpoints are saved every 5 epochs

### "Training is slow"
- Check GPU is enabled: Run `!nvidia-smi`
- Should see "Tesla T4" with ~15GB memory
- If it says "CPU", change runtime type to GPU

---

## ✅ Success Checklist

Before starting training, verify:

- [ ] Google One account has 2TB storage
- [ ] Audio files uploaded to Drive (check file count)
- [ ] Checkpoint uploaded (294MB file)
- [ ] Vocab.json uploaded
- [ ] Colab notebook opened with GPU enabled
- [ ] Code package uploaded to Colab
- [ ] All files linked/copied successfully
- [ ] Training started and showing progress

---

## 📝 Notes

- Training saves checkpoints every 5 epochs automatically
- Best model is saved when validation loss improves
- You can close the browser and come back (but keep tab open to prevent disconnect)
- Total cost: **$0** (using free Colab + your existing Google One storage)

---

Good luck with your training! 🚀
