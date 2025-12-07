# Synthetic Audio Generation for ASR

This directory contains scripts for generating synthetic Konkani audio using Text-to-Speech.

## Scripts

### `generate_synthetic_audio.py`

Converts your 47k Konkani text samples into audio files using Google Text-to-Speech (gTTS).

**Usage**:

```bash
# Test mode (10 samples only)
python generate_synthetic_audio.py --test

# Generate all samples
python generate_synthetic_audio.py

# Generate specific number
python generate_synthetic_audio.py --max-samples 1000

# Resume from specific index
python generate_synthetic_audio.py --start-from 5000
```

**Output**:
- Audio files: `data/audio/synthetic/konkani_*.wav`
- Manifest: `data/audio/audio_manifest.csv`

**Time**: 2-4 hours for full dataset (47k samples)

## Requirements

Install audio dependencies:
```bash
pip install gTTS librosa soundfile
```

## Next Steps

After generating audio:
1. Train ASR model: `src/training/train_asr_synthetic.py`
2. Test ASR: `src/inference/test_konkani_asr.py`
3. Integrate with sentiment: `src/inference/test_audio_sentiment.py`
