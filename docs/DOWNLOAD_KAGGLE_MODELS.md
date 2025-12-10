# Download Your Trained ASR Models from Kaggle

## 🎉 Congratulations! Your training is complete!

Now let's get those .pt checkpoint files to your local machine.

## Method 1: Kaggle CLI (Fastest)

### Step 1: Setup Kaggle API (One-time)

1. **Get your API token:**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New Token"
   - This downloads `kaggle.json`

2. **Install the token:**
   ```bash
   # Create kaggle directory
   mkdir -p ~/.kaggle
   
   # Move the downloaded token
   mv ~/Downloads/kaggle.json ~/.kaggle/
   
   # Set permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Verify it works:**
   ```bash
   kaggle --version
   ```

### Step 2: Download Your Models

```bash
# Download to a new folder
kaggle kernels output stavin12/final -p ./kaggle_asr_outputs

# Check what you got
ls -lh kaggle_asr_outputs/
```

## Method 2: Web UI (Easier)

1. **Go to your Kaggle notebook:**
   - https://www.kaggle.com/code/stavin12/final

2. **Click the "Output" tab** (right side of the page)

3. **Download files:**
   - Click "Download All" to get everything as a zip
   - OR click individual files to download specific checkpoints

4. **Extract the zip:**
   ```bash
   unzip ~/Downloads/final.zip -d ./kaggle_asr_outputs
   ```

## 📦 What You Should Have

After downloading, you should see:

```
kaggle_asr_outputs/
├── checkpoints/
│   ├── best_model.pt              # ⭐ Your best model!
│   ├── checkpoint_epoch_5.pt      # Checkpoint at epoch 5
│   ├── checkpoint_epoch_10.pt     # Checkpoint at epoch 10
│   ├── checkpoint_epoch_15.pt     # etc...
│   └── ...
└── logs/
    └── training_logs.txt          # Training history
```

## 🎯 Check Your Best Model

```bash
# See file sizes
ls -lh kaggle_asr_outputs/checkpoints/

# Check which epoch was best
grep "Saved best model" kaggle_asr_outputs/logs/* | tail -5
```

## 📊 Inspect Model Info

Create a quick script to check your model:

```python
import torch

# Load the best model
checkpoint = torch.load('kaggle_asr_outputs/checkpoints/best_model.pt', 
                        map_location='cpu')

print("="*60)
print("🎯 BEST MODEL INFO")
print("="*60)
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
print(f"CTC Loss: {checkpoint.get('ctc_loss', 'N/A'):.4f}")
print("="*60)

# Check model architecture
if 'model_config' in checkpoint:
    print("\n📐 Model Architecture:")
    for key, value in checkpoint['model_config'].items():
        print(f"  {key}: {value}")
```

## 🚀 Next Steps

### 1. Move to Your Project
```bash
# Copy best model to your project
cp kaggle_asr_outputs/checkpoints/best_model.pt checkpoints/

# Or copy all checkpoints
cp -r kaggle_asr_outputs/checkpoints/* checkpoints/
```

### 2. Test Your Model
```bash
# Run inference on test audio
python scripts/test_asr_model.py \
    --checkpoint checkpoints/best_model.pt \
    --audio data/test_audio.wav
```

### 3. Compare with Previous Models
```bash
# List all your checkpoints
ls -lh checkpoints/*.pt

# Compare validation losses
python scripts/compare_checkpoints.py
```

## 💡 Pro Tips

### Download Specific Files Only
```bash
# If you only want the best model
kaggle kernels output stavin12/final -p ./temp
cp ./temp/checkpoints/best_model.pt checkpoints/
rm -rf ./temp
```

### Check Training Logs
```bash
# View the last 50 lines of training
tail -50 kaggle_asr_outputs/logs/training_logs.txt

# Search for best validation loss
grep "Val Loss" kaggle_asr_outputs/logs/training_logs.txt | sort -k4 -n | head -5
```

### Backup Your Models
```bash
# Create a backup with timestamp
BACKUP_DIR="model_backups/asr_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp -r kaggle_asr_outputs/checkpoints/* $BACKUP_DIR/
echo "✅ Backed up to: $BACKUP_DIR"
```

## 🎯 Quick Download Script

Save this as `download_models.sh`:

```bash
#!/bin/bash
echo "📥 Downloading trained models from Kaggle..."

# Download
kaggle kernels output stavin12/final -p ./kaggle_asr_outputs

# Check what we got
echo ""
echo "✅ Downloaded files:"
ls -lh kaggle_asr_outputs/checkpoints/

# Copy best model
cp kaggle_asr_outputs/checkpoints/best_model.pt checkpoints/
echo ""
echo "✅ Copied best_model.pt to checkpoints/"

# Show model info
python3 << EOF
import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
print("\n🎯 Best Model Info:")
print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"  Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}")
EOF
```

Make it executable:
```bash
chmod +x download_models.sh
./download_models.sh
```

## 🔍 Troubleshooting

**"Could not find kaggle.json"**
- Follow Step 1 above to set up your API token

**"Notebook not found"**
- Check your notebook URL: `https://www.kaggle.com/code/USERNAME/NOTEBOOK-NAME`
- Use: `kaggle kernels output USERNAME/NOTEBOOK-NAME`

**"No output files"**
- Make sure your notebook has finished running
- Check the Output tab in Kaggle web UI

**Files are too large**
- Download via web UI instead
- Or download specific checkpoints only

## 📈 What's Next?

Now that you have your trained models:

1. ✅ Test on real Konkani audio
2. ✅ Evaluate Word Error Rate (WER)
3. ✅ Deploy for inference
4. ✅ Fine-tune if needed
5. ✅ Share your results!

---

**Need help testing your model?** Check out `docs/ASR_INFERENCE_GUIDE.md`
