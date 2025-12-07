# 🎙️ ASR Quick Setup Guide - You're Almost Ready!

## ✅ What You Already Have in Google Drive:

Looking at your Drive, you have:
- ✅ `checkpoint_epoch_15.pt` (293.9 MB) - Your trained checkpoint
- ✅ `konkani_project.zip` (14.27 GB) - Your full project with audio data
- ✅ `vocab.json` (6 KB) - Your vocabulary file

**Great! You're 90% done with uploads!**

---

## 🚀 Next Steps (15 minutes to start training):

### Step 1: Open Google Colab (2 minutes)

1. Go to: https://colab.research.google.com
2. Sign in with the **same Google account** that has these files
3. Click "File" → "Upload notebook"
4. Upload: `train_konkanivani_google_one.ipynb` from your Mac
5. Click "Runtime" → "Change runtime type" → Select **GPU (T4)**
6. Click "Save"

---

### Step 2: Run the Notebook Cells (10 minutes setup)

#### **Cell 1: Install Dependencies** (2 minutes)
```python
print("📦 Installing dependencies...")
!pip install -q torch torchaudio tensorboard jiwer pyyaml soundfile
print("✅ Done!")

# Check GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```
**Expected output**: "Tesla T4, 15360 MiB"

---

#### **Cell 2: Mount Drive & Find Your Files** (1 minute)
```python
from google.colab import drive
import os

drive.mount('/content/drive')

print("\n🔍 Looking for your files...\n")

# Check for your uploaded files
files_to_check = {
    'Project Zip': '/content/drive/MyDrive/konkani_project.zip',
    'Checkpoint': '/content/drive/MyDrive/checkpoint_epoch_15.pt',
    'Vocab': '/content/drive/MyDrive/vocab.json'
}

all_found = True
for name, path in files_to_check.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"✅ {name}: Found ({size_mb:.1f} MB)")
    else:
        print(f"❌ {name}: Not found at {path}")
        all_found = False

if all_found:
    print("\n✅ All files found! Ready to proceed.")
else:
    print("\n⚠️  Some files missing. Check paths above.")
```

---

#### **Cell 3: Extract Project** (3 minutes)
```python
import os

print("📂 Extracting konkani_project.zip...")
print("   This will take 2-3 minutes...\n")

# Extract to /content/
!unzip -q /content/drive/MyDrive/konkani_project.zip -d /content/

print("✅ Extraction complete!\n")

# Find where files were extracted
print("🔍 Locating project files...\n")

# Check common locations
possible_paths = [
    '/content/konkani',
    '/content/konkani_project',
    '/content'
]

project_path = None
for path in possible_paths:
    if os.path.exists(f"{path}/train_konkanivani_asr.py"):
        project_path = path
        break

if project_path:
    print(f"✅ Project found at: {project_path}")
    os.chdir(project_path)
    print(f"✅ Changed to: {os.getcwd()}\n")
    
    # Verify key files
    print("📋 Verifying files:\n")
    key_files = [
        'train_konkanivani_asr.py',
        'models/konkanivani_asr.py',
        'data/audio_processing/dataset.py',
        'data/konkani-asr-v0/splits/manifests/train.json'
    ]
    
    for f in key_files:
        if os.path.exists(f):
            print(f"✅ {f}")
        else:
            print(f"❌ {f} - NOT FOUND")
else:
    print("❌ Could not find project files!")
    print("\nSearching for train_konkanivani_asr.py...")
    !find /content -name "train_konkanivani_asr.py" -type f
```

---

#### **Cell 4: Copy Checkpoint & Vocab** (30 seconds)
```python
import os
import shutil

print("📋 Setting up checkpoint and vocab...\n")

# Create checkpoints directory
os.makedirs('/content/checkpoints', exist_ok=True)

# Copy checkpoint
checkpoint_src = '/content/drive/MyDrive/checkpoint_epoch_15.pt'
checkpoint_dst = '/content/checkpoints/checkpoint_epoch_15.pt'

if os.path.exists(checkpoint_src):
    shutil.copy(checkpoint_src, checkpoint_dst)
    size_mb = os.path.getsize(checkpoint_dst) / (1024*1024)
    print(f"✅ Copied checkpoint ({size_mb:.1f} MB)")
else:
    print(f"❌ Checkpoint not found at: {checkpoint_src}")

# Copy vocab if not already in project
vocab_src = '/content/drive/MyDrive/vocab.json'
vocab_dst = 'vocab.json'

if not os.path.exists(vocab_dst) and os.path.exists(vocab_src):
    shutil.copy(vocab_src, vocab_dst)
    print(f"✅ Copied vocab.json")
elif os.path.exists(vocab_dst):
    print(f"✅ vocab.json already present")
else:
    print(f"⚠️  vocab.json not found")

print("\n✅ Setup complete! Ready to train.")
```

---

#### **Cell 5: Start Training!** (12 hours)
```python
import os

print("="*70)
print("🚀 STARTING KONKANIVANI ASR TRAINING")
print("="*70)

# Verify checkpoint
checkpoint_path = "/content/checkpoints/checkpoint_epoch_15.pt"
if os.path.exists(checkpoint_path):
    print("\n✅ Resuming from checkpoint_epoch_15.pt")
    print("   Training: Epoch 16 → 50 (35 epochs)")
    print("   Estimated time: ~12 hours\n")
    resume_flag = f"--resume {checkpoint_path}"
else:
    print("\n⚠️  Starting from scratch")
    print("   Training: Epoch 1 → 50")
    print("   Estimated time: ~20 hours\n")
    resume_flag = ""

print("📊 Configuration:")
print("   • GPU: Tesla T4")
print("   • Batch size: 16")
print("   • Model: d_model=256, 12 encoder, 6 decoder layers")
print("   • Saves checkpoint every 5 epochs\n")

print("="*70)
print("TRAINING STARTED - KEEP THIS TAB OPEN!")
print("="*70 + "\n")

# Start training
!python3 train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file vocab.json \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 0.0005 \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6 \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    {resume_flag}
```

---

#### **Cell 6: Monitor Progress** (run anytime)
```python
print("📊 Training Status\n")
print("="*70)

# Check checkpoints
print("\n💾 Saved Checkpoints:\n")
!ls -lth checkpoints/ | head -8

# Check GPU
print("\n🔥 GPU Usage:\n")
!nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

# Check logs
print("\n📝 Recent Log:\n")
!tail -20 logs/training.log 2>/dev/null || echo "Log not created yet"
```

---

#### **Cell 7: Backup to Drive** (run every 2-3 hours)
```python
from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/content/drive/MyDrive/konkanivani_backups/backup_{timestamp}"

print(f"💾 Backing up to: {backup_path}\n")

!mkdir -p {backup_path}
!cp -r checkpoints/* {backup_path}/ 2>/dev/null
!cp -r logs {backup_path}/ 2>/dev/null

print("✅ Backup complete!")
!ls -lh {backup_path}/
```

---

#### **Cell 8: Download Final Model** (after training)
```python
from google.colab import files
from datetime import datetime

print("📦 Packaging final model...\n")

# Create package
!mkdir -p final_model
!cp checkpoints/best_model.pt final_model/
!cp vocab.json final_model/
!cp -r models final_model/
!cp inference_konkanivani.py final_model/ 2>/dev/null

# Zip it
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"konkanivani_final_{timestamp}.zip"
!zip -r {zip_name} final_model/

print(f"✅ Created: {zip_name}")
!ls -lh {zip_name}

# Save to Drive
!mkdir -p /content/drive/MyDrive/konkanivani_final_models
!cp {zip_name} /content/drive/MyDrive/konkanivani_final_models/

print(f"\n✅ Saved to Drive!")
print(f"\n📥 Downloading...")
files.download(zip_name)
```

---

## 🎯 Quick Checklist:

Before starting Cell 5, verify:
- [ ] Cell 1: GPU shows "Tesla T4"
- [ ] Cell 2: All 3 files found (✅ marks)
- [ ] Cell 3: Project extracted successfully
- [ ] Cell 4: Checkpoint copied (293.9 MB)
- [ ] Ready to start training!

---

## ⏰ Timeline:

| Step | Time |
|------|------|
| Colab setup (Cells 1-4) | 10 minutes |
| Training (Cell 5) | ~12 hours |
| **Total** | **~12 hours** |

---

## 💡 Important Tips:

1. **Keep browser tab open** - Colab disconnects if inactive
2. **Run Cell 7 every 2-3 hours** - Backup your progress
3. **Check Cell 6 occasionally** - Monitor training
4. **Don't close the tab** - Training will stop

---

## 🆘 Troubleshooting:

### "Files not found in Cell 2"
- Make sure you're signed into the correct Google account
- Check file names match exactly (case-sensitive)

### "Out of memory in Cell 5"
- Change `--batch_size 16` to `--batch_size 8`

### "Runtime disconnected"
- Just reconnect and re-run Cell 5
- It will resume from the last saved checkpoint

---

## ✅ You're Ready!

Your files are already uploaded. Just:
1. Open Colab
2. Run cells 1-5
3. Wait ~12 hours
4. Download your trained model!

Good luck! 🚀
