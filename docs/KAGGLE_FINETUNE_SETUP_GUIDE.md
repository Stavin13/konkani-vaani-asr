# Kaggle Fine-Tuning Setup Guide
## Fine-tune best_model (1).pt with KonkaniRawSpeechCorpus

### Quick Start (5 Steps)

#### Step 1: Upload Your Checkpoint to Kaggle
1. Go to Kaggle.com → Datasets → New Dataset
2. Upload `best_model (1).pt` 
3. Name it something like "konkani-asr-checkpoint"
4. Make it public or private
5. Note the dataset path (e.g., `/kaggle/input/konkani-asr-checkpoint`)

#### Step 2: Add Datasets to Your Notebook
1. Create a new Kaggle Notebook
2. Click "Add Data" on the right sidebar
3. Search and add these datasets:
   - **KonkaniRawSpeechCorpus** (the 100GB dataset you found)
   - **Your checkpoint dataset** (from Step 1)
   - **Your vocab.json** (upload as dataset or include in checkpoint)

#### Step 3: Update Configuration
In Cell 2 of the notebook, update these paths:

```python
CONFIG = {
    # Update these 3 paths:
    'checkpoint_path': '/kaggle/input/konkani-asr-checkpoint/best_model (1).pt',
    'dataset_path': '/kaggle/input/konkanirawspeechcorpus',
    'vocab_path': '/kaggle/input/konkani-asr-checkpoint/vocab.json',
    
    # Fine-tuning parameters (already optimized)
    'learning_rate': 1e-5,  # 10x lower for fine-tuning
    'num_epochs': 20,
    'batch_size': 8,
    # ... rest stays the same
}
```

#### Step 4: Enable GPU
1. Click "Settings" (right sidebar)
2. Under "Accelerator", select **GPU T4 x2** or **GPU P100**
3. Enable "Internet" if you need to install packages

#### Step 5: Run the Notebook
1. Click "Run All" or run cells sequentially
2. Monitor training progress in Cell 9
3. Download results from Cell 12

---

## What the Notebook Does

### Training Strategy
- **Loads your 26% accuracy model** from checkpoint
- **Reduces learning rate by 10x** (1e-4 → 1e-5) for stable fine-tuning
- **Streams data** from KonkaniRawSpeechCorpus (no memory issues)
- **Adds SpecAugment** for better generalization
- **Monitors WER** every epoch (not just loss)
- **Early stopping** if no improvement for 5 epochs

### Expected Timeline
- **Setup**: 5-10 minutes (loading data, model)
- **Training**: 15-20 hours for 20 epochs
- **Per epoch**: ~45-60 minutes with T4 GPU
- **Checkpoints**: Saved every 2 epochs

### Expected Results
- **Starting point**: 26% accuracy (74% WER)
- **Target**: 45-60% accuracy (40-55% WER)
- **Improvement**: +20-35% accuracy gain

---

## Dataset Information

### KonkaniRawSpeechCorpus Structure
```
/kaggle/input/konkanirawspeechcorpus/
├── splits/
│   └── manifests/
│       ├── train.json      # Training samples
│       ├── val.json        # Validation samples
│       └── test.json       # Test samples
└── data/
    └── [audio files]       # WAV/MP3 files
```

### Manifest Format
Each line in the manifest is a JSON object:
```json
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "konkani transcription",
  "duration": 5.2
}
```

---

## Troubleshooting

### Issue: "File not found" error
**Solution**: Check your paths in Cell 2. Use the exact paths from Kaggle's "Data" tab.

### Issue: Out of memory
**Solution**: Reduce batch_size in Cell 2:
```python
'batch_size': 4,  # Reduce from 8 to 4
'gradient_accumulation_steps': 8,  # Increase to maintain effective batch size
```

### Issue: Training too slow
**Solution**: 
- Use GPU T4 x2 or P100 (not CPU)
- Reduce `num_workers` to 1 in Cell 2
- Reduce `max_audio_length` to 15.0 seconds

### Issue: WER not improving
**Solution**: 
- Check if vocab.json matches your training data
- Try lower learning rate: `'learning_rate': 5e-6`
- Increase training epochs: `'num_epochs': 30`

### Issue: Checkpoint loading fails
**Solution**: Make sure your checkpoint has these keys:
```python
checkpoint = {
    'model_state_dict': ...,
    'epoch': ...,
    'val_loss': ...,
    'config': ...
}
```

---

## After Training

### Download Your Model
1. Cell 12 creates `finetuned_model.zip`
2. Download from Kaggle's "Output" tab
3. Extract and use `best_model_finetuned.pt`

### Test Your Model Locally
```python
import torch

# Load fine-tuned model
checkpoint = torch.load('best_model_finetuned.pt')
print(f"WER: {checkpoint['wer']:.2f}%")
print(f"Accuracy: {100 - checkpoint['wer']:.2f}%")

# Load into your model
model.load_state_dict(checkpoint['model_state_dict'])
```

### Compare Results
```python
# Original model
Original: 26% accuracy, 74% WER

# Fine-tuned model
Fine-tuned: {checkpoint['wer']}% WER
Improvement: {26 - (100 - checkpoint['wer'])}% gain
```

---

## Advanced Options

### Use Different Dataset Split
If you want to use more/less data:
```python
# In Cell 7, modify:
train_manifest = '/kaggle/input/konkanirawspeechcorpus/splits/manifests/train.json'

# Or create custom split:
# Use first 80% for training, 20% for validation
```

### Adjust Learning Rate Schedule
```python
# In Cell 6, modify warmup:
warmup_steps = steps_per_epoch * 5  # 5 epochs warmup instead of 2
```

### Add More Augmentation
```python
# In Cell 3, add to _extract_features:
if self.augment:
    # Add noise
    noise = torch.randn_like(waveform) * 0.005
    waveform = waveform + noise
    
    # Speed perturbation
    speed_factor = random.uniform(0.9, 1.1)
    # ... apply speed change
```

---

## Need Help?

### Check Training Progress
Monitor these metrics in Cell 9:
- **Train Loss**: Should decrease steadily
- **Val Loss**: Should decrease (may plateau)
- **WER**: Should decrease (this is what matters!)
- **Learning Rate**: Should decrease gradually

### Sample Output
```
Epoch 5/20
Train Loss: 1.8234
Val Loss: 1.9456
WER: 65.23%
✓ New best WER: 65.23%

Sample predictions:
  Ref: konkani text here
  Pred: konkani text here
```

### Good Signs
- WER decreasing every few epochs
- Sample predictions getting closer to references
- Val loss not increasing (no overfitting)

### Bad Signs
- WER increasing → learning rate too high
- Loss not changing → learning rate too low
- Val loss increasing → overfitting (stop early)

---

## Summary

You're fine-tuning a 26% accuracy model using:
- **Dataset**: KonkaniRawSpeechCorpus (100GB)
- **Method**: Streaming + lower learning rate
- **Goal**: 45-60% accuracy
- **Time**: 15-20 hours on Kaggle GPU

The notebook handles everything automatically. Just update the 3 paths and run!
