# 🚀 Quick Start Guide - 5 Steps to Training

## Step 1: Upload to Google Drive (1-2 hours)
```bash
# On your Mac
cd /Volumes/data\&proj/konkani

# Option A: Use Google Drive Desktop app (recommended)
# - Install from: https://www.google.com/drive/download/
# - Sign in with your Google One account
# - Drag these folders to your Drive:
#   • konkani-asr-v0 folder
#   • checkpoint_epoch_15.pt
#   • vocab.json

# Option B: Upload via web browser
# - Go to drive.google.com
# - Create folder: konkanivani_training
# - Upload the files above
```

## Step 2: Create Code Package (1 minute)
```bash
# On your Mac
cd /Volumes/data\&proj/konkani
./create_colab_package.sh

# This creates: konkani_code.zip (38MB)
```

## Step 3: Open Colab (1 minute)
1. Go to: https://colab.research.google.com
2. Sign in with your **Google One account**
3. Upload notebook: `train_konkanivani_google_one.ipynb`
4. Runtime → Change runtime type → **GPU (T4)**

## Step 4: Run Cells in Colab (5 minutes setup)
1. **Cell 1**: Install dependencies
2. **Cell 2**: Mount Drive & verify files
3. **Cell 3**: Upload `konkani_code.zip`
4. **Cell 4**: Link data from Drive
5. **Cell 5**: Start training ← This runs for ~12 hours

## Step 5: Keep Alive & Monitor
- Keep browser tab open
- Run Cell 6 to check progress
- Run Cell 7 every 2-3 hours to backup

---

## ⏰ Timeline
- Upload to Drive: 1-2 hours
- Colab setup: 5 minutes
- Training: ~12 hours (Epochs 16-50)
- **Total: ~14 hours**

---

## ✅ Success Checklist
Before starting Cell 5, verify:
- [ ] Cell 2 shows all files found (✅ marks)
- [ ] Cell 3 extracted code successfully
- [ ] Cell 4 linked audio files
- [ ] GPU shows "Tesla T4" in Cell 1

---

## 🆘 Need Help?
See: `GOOGLE_ONE_SETUP_GUIDE.md` for detailed instructions
