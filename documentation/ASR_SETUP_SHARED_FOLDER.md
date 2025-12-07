# 🎙️ ASR Setup - Using Shared Folder Link

## ✅ What You Have:
- Shared folder link: https://drive.google.com/drive/folders/1KX7k_z2negFKq3qFjHJh-K1U-MEcNp7P
- Contains: checkpoint_epoch_15.pt, konkani_project.zip, vocab.json
- Account B will run Colab

---

## STEP 1: Add Shared Folder to Account B's Drive (1 minute)

1. **Open the link** in Account B's browser:
   ```
   https://drive.google.com/drive/folders/1KX7k_z2negFKq3qFjHJh-K1U-MEcNp7P
   ```

2. **Sign in as Account B** if prompted

3. **Add to your Drive**:
   - Click the folder name at the top
   - Click "Add shortcut to Drive" (⭐ icon)
   - Choose "My Drive"
   - Click "Add shortcut"

4. **Verify**:
   - Go to drive.google.com
   - You should see the folder in "My Drive"
   - Open it and verify you see all 3 files

---

## STEP 2: Open Colab with Account B (2 minutes)

1. Go to: https://colab.research.google.com
2. **Sign in as Account B**
3. Click "File" → "New notebook"
4. Click "Runtime" → "Change runtime type"
5. Select **GPU** → **T4**
6. Click "Save"

---

## STEP 3: Run These Cells (Copy-Paste Each One)

### **Cell 1: Install Dependencies** ⏱️ 2 min

```python
print("📦 Installing dependencies...")
!pip install -q torch torchaudio tensorboard jiwer pyyaml soundfile
print("✅ Dependencies installed!\n")

# Check GPU
print("🔍 GPU Check:")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("\n✅ If you see 'Tesla T4, 15360 MiB' above, you're ready!")
```

---

### **Cell 2: Mount Drive & Access Shared Folder** ⏱️ 1 min

```python
from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')

print("\n" + "="*70)
print("🔍 ACCESSING SHARED FOLDER")
print("="*70 + "\n")

# The shared folder should be in My Drive after you added the shortcut
# Let's find it
print("Searching for shared folder...\n")

# Common locations for shared folders
possible_locations = [
    "/content/drive/MyDrive",
    "/content/drive/Shareddrives"
]

# Search for the files
import subprocess
result = subprocess.run(
    ["find", "/content/drive", "-name", "checkpoint_epoch_15.pt", "-o", "-name", "konkani_project.zip"],
    capture_output=True,
    text=True
)

found_files = result.stdout.strip().split('\n')
found_files = [f for f in found_files if f]  # Remove empty strings

if found_files:
    print("✅ Found files:\n")
    for f in found_files:
        print(f"   {f}")
    
    # Get the folder path
    folder_path = os.path.dirname(found_files[0])
    print(f"\n✅ Folder location: {folder_path}\n")
    
    # Verify all 3 files
    files_to_check = ['checkpoint_epoch_15.pt', 'konkani_project.zip', 'vocab.json']
    print("📋 Verifying files:\n")
    
    all_found = True
    for filename in files_to_check:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {filename} - NOT FOUND")
            all_found = False
    
    if all_found:
        # Save the folder path for next cells
        with open('/tmp/folder_path.txt', 'w') as f:
            f.write(folder_path)
        print("\n" + "="*70)
        print("✅ ALL FILES FOUND! Ready to proceed.")
        print("="*70)
    else:
        print("\n⚠️  Some files missing!")
else:
    print("❌ Could not find files!")
    print("\nTroubleshooting:")
    print("1. Did you add the shared folder shortcut to 'My Drive'?")
    print("2. Are you signed into Account B in Colab?")
    print("3. Try opening the link again and adding the shortcut")
```

---

### **Cell 3: Extract Project** ⏱️ 3 min

```python
import os

# Get folder path from previous cell
with open('/tmp/folder_path.txt', 'r') as f:
    folder_path = f.read().strip()

print(f"📂 Using files from: {folder_path}\n")
print("📦 Extracting konkani_project.zip...")
print("   This takes 2-3 minutes...\n")

# Extract
zip_path = os.path.join(folder_path, 'konkani_project.zip')
!unzip -q {zip_path} -d /content/

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
        'data/konkani-asr-v0/splits/manifests/train.json'
    ]
    
    for f in key_files:
        status = "✅" if os.path.exists(f) else "❌"
        print(f"{status} {f}")
    
    # Save project path for next cells
    with open('/tmp/project_path.txt', 'w') as f:
        f.write(project_path)
    
    print("\n✅ Ready for next step!")
else:
    print("❌ Could not find project!")
    print("\nSearching...")
    !find /content -name "train_konkanivani_asr.py" -type f 2>/dev/null
```

---

### **Cell 4: Copy Checkpoint & Vocab** ⏱️ 30 sec

```python
import os
import shutil

# Get paths from previous cells
with open('/tmp/folder_path.txt', 'r') as f:
    folder_path = f.read().strip()

with open('/tmp/project_path.txt', 'r') as f:
    project_path = f.read().strip()

os.chdir(project_path)

print("📋 Setting up checkpoint and vocab...\n")

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

# Copy checkpoint
checkpoint_src = os.path.join(folder_path, 'checkpoint_epoch_15.pt')
checkpoint_dst = 'checkpoints/checkpoint_epoch_15.pt'

if os.path.exists(checkpoint_src):
    print(f"📥 Copying checkpoint...")
    shutil.copy(checkpoint_src, checkpoint_dst)
    size_mb = os.path.getsize(checkpoint_dst) / (1024*1024)
    print(f"✅ Checkpoint ready ({size_mb:.1f} MB)")
else:
    print(f"❌ Checkpoint not found!")

# Copy vocab if needed
vocab_src = os.path.join(folder_path, 'vocab.json')
if not os.path.exists('vocab.json') and os.path.exists(vocab_src):
    shutil.copy(vocab_src, 'vocab.json')
    print(f"✅ Copied vocab.json")
elif os.path.exists('vocab.json'):
    print(f"✅ vocab.json already present")
else:
    print(f"⚠️  vocab.json not found")

print("\n" + "="*70)
print("✅ SETUP COMPLETE! Ready to train.")
print("="*70)
```

---

### **Cell 5: Start Training!** ⏱️ ~12 hours

```python
import os

print("="*70)
print("🚀 STARTING KONKANIVANI ASR TRAINING")
print("="*70)

# Verify checkpoint
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
print("   • GPU: Tesla T4")
print("   • Batch size: 16")
print("   • Model: d_model=256, 12 encoder, 6 decoder layers")
print("   • Checkpoints: Every 5 epochs")
print("   • Data: From shared folder\n")

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

### **Cell 6: Monitor Progress** (run anytime)

```python
from datetime import datetime

print(f"📊 Training Status - {datetime.now().strftime('%H:%M:%S')}\n")
print("="*70)

# Checkpoints
print("\n💾 Saved Checkpoints:\n")
!ls -lth checkpoints/ 2>/dev/null | head -8 || echo "No checkpoints yet"

# GPU
print("\n🔥 GPU Usage:\n")
!nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

# Logs
print("\n📝 Recent Training Log:\n")
!tail -25 logs/training.log 2>/dev/null || echo "Log not created yet"

print("\n" + "="*70)
```

---

### **Cell 7: Backup to Account B's Drive** (run every 2-3 hours)

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/content/drive/MyDrive/konkanivani_backups/backup_{timestamp}"

print(f"💾 Backing up to Account B's Drive...\n")

!mkdir -p {backup_path}
!cp -r checkpoints/* {backup_path}/ 2>/dev/null
!cp -r logs {backup_path}/ 2>/dev/null

print("✅ Backup complete!")
!ls -lh {backup_path}/

print("\n💡 Run this every 2-3 hours to save progress!")
```

---

### **Cell 8: Download Final Model** (after training)

```python
from google.colab import files
from datetime import datetime

print("📦 Packaging final model...\n")

!mkdir -p final_model
!cp checkpoints/best_model.pt final_model/ 2>/dev/null || cp checkpoints/checkpoint_epoch_50.pt final_model/best_model.pt
!cp vocab.json final_model/
!cp -r models final_model/
!cp inference_konkanivani.py final_model/ 2>/dev/null

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"konkanivani_final_{timestamp}.zip"
!zip -r {zip_name} final_model/

print(f"✅ Created: {zip_name}")
!ls -lh {zip_name}

# Save to Drive
!mkdir -p /content/drive/MyDrive/konkanivani_final_models
!cp {zip_name} /content/drive/MyDrive/konkanivani_final_models/

print(f"\n✅ Saved to Drive!")

# Download
print(f"\n📥 Downloading...")
files.download(zip_name)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
```

---

## ✅ Quick Checklist:

### Before Starting:
- [ ] Opened shared folder link in Account B's browser
- [ ] Added shortcut to "My Drive"
- [ ] Verified 3 files are visible in Drive
- [ ] Opened Colab with Account B
- [ ] Enabled GPU (T4)

### During Setup (Cells 1-4):
- [ ] Cell 1: GPU shows "Tesla T4"
- [ ] Cell 2: All 3 files found
- [ ] Cell 3: Project extracted
- [ ] Cell 4: Checkpoint copied

### During Training:
- [ ] Cell 5: Training started
- [ ] Keep tab open
- [ ] Run Cell 7 every 2-3 hours
- [ ] Check Cell 6 occasionally

---

## 🆘 If Cell 2 Can't Find Files:

Try this alternative in Cell 2:

```python
# Manual search
print("Searching entire Drive...\n")
!find /content/drive -name "checkpoint_epoch_15.pt" 2>/dev/null
!find /content/drive -name "konkani_project.zip" 2>/dev/null
!find /content/drive -name "vocab.json" 2>/dev/null
```

Then update the `folder_path` in Cell 3 based on where files are found.

---

## ⏰ Timeline:

| Step | Time |
|------|------|
| Add folder to Drive | 1 min |
| Colab setup (Cells 1-4) | 7 min |
| Training (Cell 5) | ~12 hours |
| **Total** | **~12 hours** |

---

You're ready! Just add the shared folder to Account B's Drive and run the cells. 🚀
