# Google Colab Training Setup Guide

## Why Use Colab?

**Speed Comparison:**
- **Your Mac**: ~4 sec/batch, 75 hours total (3+ days)
- **Colab Free (T4 GPU)**: ~0.5 sec/batch, 10-15 hours total
- **Colab Pro (A100 GPU)**: ~0.2 sec/batch, 4-6 hours total

**Benefits:**
- 8-15x faster training
- Larger batch sizes (better model quality)
- Full CUDA support (no CPU fallback)
- Can use bigger model (15M params vs 9M)

---

## Step-by-Step Setup

### 1. Prepare Your Dataset

**Option A: Upload to Google Drive** (Recommended for 16GB dataset)

```bash
# On your Mac, create a zip of essential files
cd /Volumes/data\&proj/konkani
zip -r konkani_project.zip \
    models/ \
    data/audio_processing/ \
    data/konkani-asr-v0/ \
    train_konkanivani_asr.py \
    inference_konkanivani.py \
    evaluate_konkanivani.py \
    vocab.json
```

Then:
1. Upload `konkani_project.zip` to your Google Drive
2. Or use Google Drive Desktop to sync the entire `konkani` folder

**Option B: Use Cloud Storage** (Faster)

If your dataset is already on Hugging Face, AWS S3, or similar:
```python
# In Colab, download directly
!wget https://your-dataset-url/konkani-asr-v0.zip
!unzip konkani-asr-v0.zip
```

---

### 2. Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `train_konkanivani_colab.ipynb` (created in your project)

---

### 3. Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Choose **T4 GPU** (free tier) or **A100** (Pro)
4. Click **Save**

---

### 4. Run the Notebook

Execute cells in order:

**Cell 1**: Check GPU
```python
!nvidia-smi
```
You should see GPU info (T4, V100, or A100)

**Cell 2**: Install dependencies
```python
!pip install -q torch torchaudio tensorboard jiwer pyyaml soundfile
```

**Cell 3**: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 4**: Copy your project
```python
!cp -r /content/drive/MyDrive/konkani /content/konkanivani
%cd /content/konkanivani
```

**Cell 5**: Start training
```python
!python3 train_konkanivani_asr.py \
    --batch_size 16 \
    --num_epochs 50 \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6
```

---

### 5. Monitor Training

**Option A: In Colab**
```python
%load_ext tensorboard
%tensorboard --logdir logs
```

**Option B: Check progress**
```python
!tail -20 logs/events.out.tfevents.*
```

---

### 6. Download Results

After training completes:

```python
# Zip checkpoints
!zip -r checkpoints.zip checkpoints/

# Download
from google.colab import files
files.download('checkpoints.zip')
```

Or save to Drive:
```python
!cp -r checkpoints /content/drive/MyDrive/konkanivani_checkpoints
```

---

## Training Configuration

### Free Tier (T4 GPU)
```python
--batch_size 16
--d_model 256
--encoder_layers 12
--decoder_layers 6
```
**Time**: ~10-15 hours

### Pro Tier (A100 GPU)
```python
--batch_size 32
--d_model 256
--encoder_layers 12
--decoder_layers 6
```
**Time**: ~4-6 hours

### If Out of Memory
Reduce batch size:
```python
--batch_size 8
```

---

## Tips

1. **Keep Colab Active**: Colab disconnects after ~90 min of inactivity
   - Install browser extension: "Colab Auto Clicker"
   - Or run this in a cell:
   ```javascript
   function KeepAlive() {
     console.log("Keeping alive");
     document.querySelector("colab-connect-button").click();
   }
   setInterval(KeepAlive, 60000);
   ```

2. **Save Checkpoints Frequently**: 
   - Training saves every 5 epochs
   - Copy to Drive after each checkpoint

3. **Resume Training**: If disconnected, restart from last checkpoint:
   ```python
   !python3 train_konkanivani_asr.py \
       --resume checkpoints/checkpoint_epoch_10.pt \
       --batch_size 16 \
       --device cuda
   ```

4. **Monitor GPU Usage**:
   ```python
   !nvidia-smi -l 5  # Update every 5 seconds
   ```

---

## Troubleshooting

**"Out of Memory"**
- Reduce `--batch_size` to 8 or 4
- Reduce `--d_model` to 192

**"Runtime disconnected"**
- Save checkpoints to Drive regularly
- Use Colab Pro for longer sessions (24h vs 12h)

**"Files not found"**
- Check Drive mount: `!ls /content/drive/MyDrive/`
- Verify paths in manifest files

---

## After Training

1. Download `best_model.pt` from checkpoints
2. Copy back to your Mac
3. Run inference locally:
   ```bash
   python3 inference_konkanivani.py \
       --checkpoint checkpoints/best_model.pt \
       --vocab vocab.json \
       --audio test_audio.wav \
       --device mps
   ```

---

## Cost Comparison

| Option | GPU | Time | Cost |
|--------|-----|------|------|
| Mac M-series | MPS | 75h | Free (electricity) |
| Colab Free | T4 | 15h | Free |
| Colab Pro | A100 | 6h | $10/month |
| Colab Pro+ | A100 | 4h | $50/month |

**Recommendation**: Start with Colab Free (T4). If you need faster results, upgrade to Pro for one month.
