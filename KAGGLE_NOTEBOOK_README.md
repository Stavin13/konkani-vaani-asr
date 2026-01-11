# 📓 Ready-to-Use Kaggle Fine-tuning Notebook

## What is this?

This is a **complete, ready-to-use Kaggle notebook** that automatically handles both:
- ✅ **Fine-tuning** from your `best_model.pt` checkpoint
- ✅ **Training from scratch** if no checkpoint is provided

## 🚀 Quick Start (3 Steps)

### Step 1: Upload the Notebook to Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Click **"File"** → **"Import Notebook"**
4. Upload `kaggle-finetuning-notebook.ipynb`

### Step 2: Add Your Datasets

Click **"Add Data"** and add these 3 datasets:

1. **Your checkpoint** (if fine-tuning):
   - Upload `best_model (1).pt` as a new dataset
   - OR add existing checkpoint dataset
   - Name it something like `konkani-asr-checkpoint`

2. **Training data**:
   - Add your `konkani-asr-complete-data` dataset
   
3. **Training scripts**:
   - Add your `kaggle-training-scripts` dataset

### Step 3: Update Paths and Run

1. **Update checkpoint path** in the notebook (Cell 4):
   ```python
   CHECKPOINT_PATH = '/kaggle/input/YOUR-CHECKPOINT-DATASET/best_model (1).pt'
   ```

2. **Update data paths** if needed (Cell 7):
   ```python
   DATA_ROOT = Path('/kaggle/input/YOUR-DATA-DATASET')
   SCRIPTS_ROOT = Path('/kaggle/input/YOUR-SCRIPTS-DATASET')
   ```

3. Click **"Run All"** and let it train!

## 🎯 What Happens Automatically

### If Checkpoint is Found:
```
🔄 FINE-TUNING MODE ACTIVATED
   - Learning rate: 0.00001 (10x smaller)
   - Epochs: 50 (fewer epochs)
   - Loads pre-trained weights
   - Continues improving from checkpoint
```

### If No Checkpoint:
```
🆕 TRAINING FROM SCRATCH MODE
   - Learning rate: 0.0001 (standard)
   - Epochs: 100 (full training)
   - Random initialization
   - Standard training
```

## 📊 Features

✅ **Automatic Mode Detection**: Detects checkpoint and adjusts settings
✅ **Multi-GPU Support**: Automatically uses all available GPUs
✅ **Custom Vocabulary**: Generates vocab from your training data
✅ **Progress Tracking**: Shows detailed progress and statistics
✅ **Checkpoint Saving**: Saves every 5 epochs + best model
✅ **Error Handling**: Falls back to training from scratch if checkpoint fails

## 📁 What You'll Get

After training completes:

```
/kaggle/working/checkpoints/
├── best_model.pt              ← Your fine-tuned model!
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
└── ...

/kaggle/working/logs/
└── tensorboard logs
```

## 🔧 Customization

### Change Learning Rate

In Cell 13 (training config), modify:
```python
if RESUME_TRAINING:
    learning_rate = 0.000005  # Even smaller for fine-tuning
    num_epochs = 30           # Even fewer epochs
```

### Change Model Architecture

In Cell 13, modify the `config['model']` section:
```python
'model': {
    'd_model': 256,        # Larger model
    'encoder_layers': 12,  # More layers
    'dropout': 0.4,        # Higher dropout
    # ...
}
```

### Change Batch Size

In Cell 13:
```python
'batch_size': 4,  # Smaller if GPU memory issues
```

## 📋 Checklist Before Running

- [ ] Notebook uploaded to Kaggle
- [ ] Checkpoint dataset added (if fine-tuning)
- [ ] Data dataset added
- [ ] Scripts dataset added
- [ ] `CHECKPOINT_PATH` updated in Cell 4
- [ ] `DATA_ROOT` and `SCRIPTS_ROOT` updated in Cell 7
- [ ] GPU accelerator enabled (Settings → Accelerator → GPU T4 x2)

## 🎓 Understanding the Output

### Good Signs (Fine-tuning Working):
```
Epoch 1/50
  Train Loss: 2.234  ← Low, close to checkpoint's loss
  Val Loss: 2.189    ← Gradually decreasing
  
✅ Saved best model with val_loss: 2.145
```

### Warning Signs:
```
Epoch 1/50
  Train Loss: 12.456  ← Very high = weights didn't load!
  Val Loss: 11.234
```

If you see high initial loss, check:
1. Checkpoint path is correct
2. Vocab size matches
3. No error messages in checkpoint loading cell

## 🐛 Troubleshooting

### "Checkpoint not found"
**Fix**: Check the path
```python
# In a new cell, run:
!ls -lh /kaggle/input/
!ls -lh /kaggle/input/your-checkpoint-dataset/
```

### "RuntimeError: size mismatch"
**Fix**: Vocab size changed. Make sure you're using the same vocab.json from original training.

### "CUDA out of memory"
**Fix**: Reduce batch size in Cell 13:
```python
'batch_size': 4,  # Reduce from 8
```

### Loss increases during fine-tuning
**Fix**: Learning rate too high. In Cell 13:
```python
learning_rate = 0.000005  # Even smaller
```

## 📥 Download Your Model

After training, the notebook automatically shows a download link.

Or manually download:
```python
from IPython.display import FileLink
FileLink('/kaggle/working/checkpoints/best_model.pt')
```

## 🔄 Iterating

To fine-tune again:
1. Download the `best_model.pt` from this run
2. Upload it as a new checkpoint dataset
3. Run the notebook again with the new checkpoint

## 💡 Pro Tips

1. **Monitor progress**: Watch the validation loss - it should decrease steadily
2. **Save intermediate checkpoints**: The notebook saves every 5 epochs
3. **Compare results**: Note the final val loss and compare with original checkpoint
4. **Test on held-out data**: Don't rely only on validation loss

## 📞 Need Help?

If something doesn't work:
1. Check the error messages carefully
2. Verify all paths are correct
3. Make sure datasets are properly added
4. Check GPU is enabled in Kaggle settings
5. Review the troubleshooting section above

## 📚 Related Files

- `FINETUNING_GUIDE.md` - Detailed explanation of fine-tuning concepts
- `FINETUNING_QUICK_START.md` - Quick summary of changes
- `NOTEBOOK_CHANGES_GUIDE.md` - Side-by-side comparison
- `kaggle_finetuning_cells.md` - Individual cells to copy-paste

## ✨ That's It!

You now have a complete, production-ready notebook that:
- Automatically detects and loads checkpoints
- Adjusts hyperparameters for fine-tuning
- Handles errors gracefully
- Saves your progress regularly
- Works with multi-GPU setups

Just upload, configure paths, and run! 🚀
