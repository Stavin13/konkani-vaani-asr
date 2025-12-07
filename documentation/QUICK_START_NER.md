# Quick Start: NER Training on Colab

Since your Mac has path issues with the `&` character, let's train NER on Google Colab instead!

---

## 🚀 Steps (10 minutes setup, then 2-3 hours training)

### **Step 1: Open Colab**
1. Go to https://colab.research.google.com
2. Click "File" → "Upload notebook"
3. Upload `train_ner_colab.ipynb` from your project

### **Step 2: Enable GPU**
1. Click "Runtime" → "Change runtime type"
2. Hardware accelerator: **GPU**
3. GPU type: **T4**
4. Click "Save"

### **Step 3: Prepare Files to Upload**

On your Mac, create a zip with these files:

```bash
# In Terminal (from your project directory)
zip -r ner_files.zip \
    transcripts_konkani_cleaned.json \
    scripts/auto_label_ner.py \
    models/konkani_ner.py \
    train_konkani_ner.py
```

This creates `ner_files.zip` (~16MB)

### **Step 4: Run Colab Cells**

Run cells in order:

1. **Cell 1:** Install dependencies (1 min)
2. **Cell 2:** Mount Google Drive (click authorize)
3. **Cell 3:** Upload `ner_files.zip` (2 min)
4. **Cell 4:** Auto-label data (15 min) ⏰
5. **Cell 5:** Train model (2-3 hours) ⏰
6. **Cell 6:** Check results
7. **Cell 7:** Test model
8. **Cell 8:** Backup to Drive
9. **Cell 9:** Download model

---

## ⏱️ Timeline

| Time | Task | Status |
|------|------|--------|
| Now | Setup Colab | 5 min |
| +5 min | Upload files | 2 min |
| +7 min | Auto-label | 15 min |
| +22 min | Train NER | 2-3 hours |
| +3 hours | Done! | ✅ |

**While NER trains on Colab, your ASR continues training on the other Colab session!**

---

## 🎯 Benefits of Colab

✅ **Faster** - GPU is faster than Mac MPS
✅ **No path issues** - Clean environment
✅ **Parallel** - Run alongside ASR training
✅ **Free** - No cost

---

## 📊 Expected Results

After training completes:

```
checkpoints/ner/
├── best_ner_model.pt          # Your trained model (~50MB)
├── vocabularies.json          # Word/char vocabularies
└── ner_checkpoint_epoch_*.pt  # Checkpoints

Training metrics:
  • Final F1 Score: 0.75-0.80
  • Training time: ~2-3 hours
  • Model size: ~2-3M parameters
```

---

## 🐛 If Something Goes Wrong

### "Out of memory"
In Cell 5, change `--batch_size 32` to `--batch_size 16`

### "Runtime disconnected"
Just reconnect and re-run Cell 5 (training will resume from checkpoint)

### "Files not found"
Make sure you extracted `ner_files.zip` in Cell 3

---

## ✅ After Training

You'll have:
1. ✅ Trained NER model
2. ✅ Backed up to Google Drive
3. ✅ Downloaded to your computer

**Ready to integrate into your complete system on Monday!**

---

**Start now:** Upload `train_ner_colab.ipynb` to Colab and begin! 🚀
