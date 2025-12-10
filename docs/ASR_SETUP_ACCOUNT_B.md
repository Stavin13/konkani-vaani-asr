# 🎙️ ASR Setup - Using Account B (Different Gmail)

## 📋 Your Situation:
- **Account A**: Has the files in Drive (checkpoint, project zip, vocab)
- **Account B**: Will run Colab (Account A hit usage limit)

---

## PART 1: Share Files from Account A to Account B (5 minutes)

### Step 1: Sign into Account A

1. Go to: https://drive.google.com
2. Sign in as **Account A** (the one with the files)

### Step 2: Share Each File with Account B

**For `checkpoint_epoch_15.pt`:**
1. Find the file in Drive
2. Right-click → "Share"
3. In "Add people and groups", enter **Account B's email**
4. Change permission from "Viewer" to **"Editor"**
5. Click "Send"

**Repeat for:**
- `konkani_project.zip`
- `vocab.json`

**You should send 3 share invitations total.**

---

## PART 2: Accept Shares in Account B (2 minutes)

### Step 1: Check Email

1. Open email for **Account B**
2. You'll see 3 emails from Google Drive
3. Click "Open" in each email

### Step 2: Add to Your Drive

1. Go to: https://drive.google.com (sign in as **Account B**)
2. Click "Shared with me" in left sidebar
3. You should see:
   - checkpoint_epoch_15.pt
   - konkani_project.zip
   - vocab.json

4. **Select all 3 files** (hold Shift and click)
5. Right-click → "Organize" → "Add shortcut"
6. Choose "My Drive"
7. Click "Add"

### Step 3: Verify

1. Click "My Drive" in left sidebar
2. You should now see all 3 files there
3. ✅ Ready for Colab!

---

## PART 3: Run Colab with Account B (10 minutes setup)

### Step 1: Open Colab

1. Go to: https://colab.research.google.com
2. **Sign in as Account B** (important!)
3. Click "File" → "New notebook"
4. Click "Runtime" → "Change runtime type" → **GPU (T4)** → "Save"

---

### Step 2: Run These Cells in Order

#### **Cell 1: Install Dependencies** ⏱️ 2 min
```python
print("📦 Installing dependencies...")
!pip install -q torch torchaudio tensorboard jiwer pyyaml soundfile
print("✅ Done!\n")

# Verify GPU
print("🔍 Checking GPU...")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("\n✅ If you see 'Tesla T4' above, you're good!")
```

---

#### **Cell 2: Mount Drive & Find Shared Files** ⏱️ 1 min
```python
from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')

print("\n" + "="*70)
print("🔍 LOOKING FOR SHARED FILES FROM ACCOUNT A")
print("="*70 + "\n")

# Check in My Drive (where you added shortcuts)
base_path = "/content/drive/MyDrive"

files_to_check = {
    'Project Zip': f'{base_path}/konkani_project.zip',
    'Checkpoint': f'{base_path}/checkpoint_epoch_15.pt',
    'Vocab': f'{base_path}/vocab.json'
}

all_found = True
for name, path in files_to_check.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"✅ {name}: Found ({size_mb:.1f} MB)")
    else:
        print(f"❌ {name}: NOT FOUND")
        print(f"   Expected at: {path}")
        all_found = False

if all_found:
    print("\n" + "="*70)
    print("✅ ALL FILES FOUND! Ready to proceed.")
    print("="*70)
else:
    print("\n" + "="*70)
    print("❌ SOME FILES MISSING!")
    print("="*70)
    print("\nTroubleshooting:")
    print("1. Make sure you added shortcuts to 'My Drive' in Account B")
    print("2. Check 'Shared with me' in Drive")
    print("3. Verify Account A shared the files with Account B's email")
    print("\nSearching for files...")
    !find /content/drive -name "*.pt" -o -name "konkani_project.zip" -o -name "vocab.json" 2>/dev/null | head -10
```

**⚠️ IMPORTANT**: If files are NOT FOUND, check the search results at the bottom and update paths in next cells.

---

#### **Cell 3: Extract Project** ⏱️ 3 min
```python
import os

print("📂 Extracting konkani_project.zip...")
print("   This takes 2-3 minutes...\n")

# Extract
!unzip -q /content/drive/MyDrive/konkani_project.zip -d /content/

print("✅ Extraction complete!\n")

# Find project location
print("🔍 Locating project files...\n")

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
    print(f"✅ Working directory: {os.getcwd()}\n")
    
    # Verify key files
    print("📋 Verifying extracted files:\n")
    key_files = [
        'train_konkanivani_asr.py',
        'models/konkanivani_asr.py',
        'data/audio_processing/dataset.py',
        'data/konkani-asr-v0/splits/manifests/train.json',
        'vocab.json'
    ]
    
    for f in key_files:
        status = "✅" if os.path.exists(f) else "❌"
        print(f"{status} {f}")
    
    print("\n✅ Ready for next step!")
else:
    print("❌ Could not find project!")
    print("\nSearching...")
    !find /content -name "train_konkanivani_asr.py" -type f 2>/dev/null
```

---

#### **Cell 4: Copy Checkpoint** ⏱️ 30 sec
```python
import os
import shutil

print("📋 Setting up checkpoint...\n")

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

# Copy checkpoint from Drive
checkpoint_src = '/content/drive/MyDrive/checkpoint_epoch_15.pt'
checkpoint_dst = 'checkpoints/checkpoint_epoch_15.pt'

if os.path.exists(checkpoint_src):
    print(f"📥 Copying checkpoint from Drive...")
    shutil.copy(checkpoint_src, checkpoint_dst)
    size_mb = os.path.getsize(checkpoint_dst) / (1024*1024)
    print(f"✅ Checkpoint ready ({size_mb:.1f} MB)")
else:
    print(f"❌ Checkpoint not found at: {checkpoint_src}")
    print("\nSearching for checkpoint...")
    !find /content/drive -name "checkpoint_epoch_15.pt" 2>/dev/null

# Verify vocab exists
if os.path.exists('vocab.json'):
    print(f"✅ vocab.json ready")
else:
    print(f"⚠️  vocab.json not found, checking Drive...")
    vocab_src = '/content/drive/MyDrive/vocab.json'
    if os.path.exists(vocab_src):
        shutil.copy(vocab_src, 'vocab.json')
        print(f"✅ Copied vocab.json from Drive")

print("\n" + "="*70)
print("✅ SETUP COMPLETE! Ready to train.")
print("="*70)
```

---

#### **Cell 5: Start Training!** ⏱️ ~12 hours
```python
import os

print("="*70)
print("🚀 STARTING KONKANIVANI ASR TRAINING")
print("="*70)

# Check for checkpoint
checkpoint_path = "checkpoints/checkpoint_epoch_15.pt"
if os.path.exists(checkpoint_path):
    print("\n✅ Resuming from checkpoint_epoch_15.pt")
    print("   Training: Epoch 16 → 50 (35 epochs)")
    print("   Estimated time: ~12 hours")
    print("   Using Account B's Colab quota\n")
    resume_flag = f"--resume {checkpoint_path}"
else:
    print("\n⚠️  Starting from scratch")
    print("   Training: Epoch 1 → 50")
    print("   Estimated time: ~20 hours\n")
    resume_flag = ""

print("📊 Configuration:")
print("   • GPU: Tesla T4 (Account B)")
print("   • Batch size: 16")
print("   • Model: d_model=256, 12 encoder, 6 decoder layers")
print("   • Checkpoints: Saved every 5 epochs")
print("   • Data: From Account A (shared)\n")

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
from datetime import datetime

print(f"📊 Training Status - {datetime.now().strftime('%H:%M:%S')}\n")
print("="*70)

# Checkpoints
print("\n💾 Saved Checkpoints:\n")
!ls -lth checkpoints/ 2>/dev/null | head -8 || echo "No checkpoints yet"

# GPU usage
print("\n🔥 GPU Usage:\n")
!nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

# Recent logs
print("\n📝 Recent Training Log:\n")
!tail -25 logs/training.log 2>/dev/null || echo "Log file not created yet"

print("\n" + "="*70)
```

---

#### **Cell 7: Backup to Account B's Drive** (run every 2-3 hours)
```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/content/drive/MyDrive/konkanivani_backups/backup_{timestamp}"

print(f"💾 Backing up to Account B's Drive...\n")
print(f"Location: {backup_path}\n")

!mkdir -p {backup_path}
!cp -r checkpoints/* {backup_path}/ 2>/dev/null
!cp -r logs {backup_path}/ 2>/dev/null

print("✅ Backup complete!\n")
print("📋 Backed up files:\n")
!ls -lh {backup_path}/

print("\n💡 Tip: Run this cell every 2-3 hours to save progress!")
```

---

#### **Cell 8: Download Final Model** (after training completes)
```python
from google.colab import files
from datetime import datetime

print("📦 Packaging final model...\n")

# Create package
!mkdir -p final_model
!cp checkpoints/best_model.pt final_model/ 2>/dev/null || cp checkpoints/checkpoint_epoch_50.pt final_model/best_model.pt
!cp vocab.json final_model/
!cp -r models final_model/
!cp inference_konkanivani.py final_model/ 2>/dev/null

# Zip
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"konkanivani_final_{timestamp}.zip"
!zip -r {zip_name} final_model/

print(f"✅ Created: {zip_name}")
!ls -lh {zip_name}

# Save to Account B's Drive
!mkdir -p /content/drive/MyDrive/konkanivani_final_models
!cp {zip_name} /content/drive/MyDrive/konkanivani_final_models/

print(f"\n✅ Saved to Account B's Drive!")
print(f"   Location: MyDrive/konkanivani_final_models/{zip_name}")

# Download to your Mac
print(f"\n📥 Downloading to your computer...")
files.download(zip_name)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
```

---

## ✅ Complete Checklist:

### Before Starting:
- [ ] Shared 3 files from Account A to Account B
- [ ] Added shortcuts to Account B's "My Drive"
- [ ] Opened Colab with Account B
- [ ] Enabled GPU (T4)

### During Setup (Cells 1-4):
- [ ] Cell 1: GPU shows "Tesla T4"
- [ ] Cell 2: All 3 files found (✅ marks)
- [ ] Cell 3: Project extracted successfully
- [ ] Cell 4: Checkpoint copied (293.9 MB)

### During Training:
- [ ] Cell 5: Training started
- [ ] Keep browser tab open
- [ ] Run Cell 7 every 2-3 hours (backup)
- [ ] Check Cell 6 occasionally (monitor)

### After Training:
- [ ] Cell 8: Download final model
- [ ] Verify model works on your Mac

---

## 🆘 Troubleshooting:

### "Files not found in Cell 2"
**Solution**: Files might be in "Shared with me" instead of "My Drive"

Try this in Cell 2:
```python
# Check Shared with me
!ls -lh "/content/drive/Shareddrives/" 2>/dev/null
!find /content/drive -name "checkpoint_epoch_15.pt" 2>/dev/null
```

Then update paths in Cells 3 and 4 based on where files are found.

### "Account B also hit limit"
- Wait 24 hours for quota reset
- Or try Kaggle (free alternative): https://kaggle.com

### "Out of memory"
- Change `--batch_size 16` to `--batch_size 8` in Cell 5

---

## 💡 Key Points:

1. **Account A**: Only stores files (no Colab usage)
2. **Account B**: Runs training (uses its Colab quota)
3. **Shared files**: Account B can read Account A's files
4. **Backups**: Saved to Account B's Drive
5. **Final model**: Downloaded to your Mac

---

You're ready! Start with Part 1 (sharing files), then run the Colab cells. 🚀
