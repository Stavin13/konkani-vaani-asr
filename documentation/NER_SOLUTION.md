# ✅ NER Training Solution - Use Colab!

## 🚨 Problem
Your Mac has path issues due to the `&` character in `/Volumes/data&proj/konkani`

## ✅ Solution
Train NER on Google Colab (same as ASR) - faster and no path issues!

---

## 📋 What You Need to Do (Right Now)

### **1. Prepare files (2 minutes)**

Run this in your Terminal:

```bash
cd /Volumes/data\&proj/konkani
bash prepare_ner_upload.sh
```

This creates `ner_files.zip` with everything needed.

---

### **2. Open Colab (1 minute)**

1. Go to: https://colab.research.google.com
2. Click "File" → "Upload notebook"
3. Upload: `train_ner_colab.ipynb`

---

### **3. Enable GPU (30 seconds)**

1. Click "Runtime" → "Change runtime type"
2. Select: **GPU** (T4)
3. Click "Save"

---

### **4. Run the notebook (3 hours total)**

Just run cells 1-9 in order:

| Cell | Task | Time |
|------|------|------|
| 1 | Install packages | 1 min |
| 2 | Mount Drive | 30 sec |
| 3 | Upload ner_files.zip | 2 min |
| 4 | Auto-label data | 15 min |
| 5 | Train model | 2-3 hours |
| 6-9 | Test & download | 5 min |

**That's it!** Just click "Run" on each cell.

---

## 🎯 Why This is Better

| Mac (broken) | Colab (works!) |
|--------------|----------------|
| ❌ Path issues | ✅ Clean environment |
| ❌ Slower (MPS) | ✅ Faster (GPU) |
| ❌ Manual setup | ✅ One-click setup |
| ⏰ 4-5 hours | ⏰ 2-3 hours |

---

## 📊 What You'll Get

After 3 hours, you'll have:

```
✅ best_ner_model.pt (trained model)
✅ vocabularies.json (word/char mappings)
✅ ner_labeled_data.json (training data)
✅ Backed up to Google Drive
✅ Downloaded to your computer
```

---

## 🔄 Parallel Training

**Right now you have:**
- 🔄 ASR training on Colab (Account 1) - Epochs 16-50
- 🔄 NER training on Colab (Account 2) - Epochs 1-20

**Both will finish by tonight!** 🎉

---

## ⏰ Timeline

**Now (8:00 PM):**
- Run `prepare_ner_upload.sh`
- Upload to Colab
- Start training

**8:15 PM:**
- Auto-labeling running

**8:30 PM:**
- Training started
- Go do something else!

**11:00 PM:**
- Both ASR and NER complete! ✅

**Monday:**
- Add emotion model
- Integrate everything
- Deploy to Hugging Face

---

## 🚀 Start Now!

```bash
# Step 1: Prepare files
cd /Volumes/data\&proj/konkani
bash prepare_ner_upload.sh

# Step 2: Open browser
# Go to: https://colab.research.google.com
# Upload: train_ner_colab.ipynb
# Upload: ner_files.zip (in Cell 3)
# Run all cells!
```

---

## 💡 Pro Tip

Keep both Colab tabs open:
- **Tab 1:** ASR training (your existing session)
- **Tab 2:** NER training (new session)

Both will run in parallel! 🚀

---

**You're almost done! Just 3 hours of training left, then you have 2/4 models complete!** 💪
