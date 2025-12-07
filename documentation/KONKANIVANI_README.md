# KonkaniVani - Custom Konkani ASR Model

**A custom Transformer-based Automatic Speech Recognition system for Konkani language.**

## 🎯 Overview

KonkaniVani is a state-of-the-art ASR model built from scratch specifically for Konkani (Devanagari script). It uses a Conformer encoder with Transformer decoder architecture and hybrid CTC+Attention training.

### Model Architecture
- **Encoder**: 12-layer Conformer (multi-head attention + convolution)
- **Decoder**: 6-layer Transformer with cross-attention
- **Loss**: Hybrid CTC (30%) + Attention (70%)
- **Vocabulary**: 200 Konkani Devanagari characters
- **Parameters**: ~15M trainable parameters

### Dataset
- **Training**: 2,549 segments (~20 hours)
- **Validation**: 318 segments (~2.5 hours)
- **Test**: 320 segments (~2.5 hours)
- **Total**: 3,187 segments (25.91 hours)

## 📁 Project Structure

```
konkani/
├── models/
│   └── konkanivani_asr.py          # Model architecture
├── data/
│   └── audio_processing/
│       ├── audio_processor.py       # Mel-spectrogram extraction
│       ├── text_tokenizer.py        # Konkani tokenizer
│       └── dataset.py               # PyTorch dataset
├── config/
│   └── model_config.yaml            # Hyperparameters
├── train_konkanivani_asr.py         # Training script
├── inference_konkanivani.py         # Inference script
├── evaluate_konkanivani.py          # Evaluation script
└── vocab.json                       # Vocabulary file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchaudio tensorboard jiwer pyyaml
```

### 2. Train the Model

```bash
python3 train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file vocab.json \
    --batch_size 16 \
    --num_epochs 50 \
    --device mps  # or 'cuda' or 'cpu'
```

**Training time**: ~12-24 hours on Apple Silicon (M1/M2/M3)

### 3. Monitor Training

```bash
tensorboard --logdir logs
```

### 4. Evaluate Model

```bash
python3 evaluate_konkanivani.py \
    --checkpoint checkpoints/best_model.pt \
    --test_manifest data/konkani-asr-v0/splits/manifests/test.json \
    --vocab_file vocab.json \
    --device mps
```

### 5. Transcribe Audio

```bash
# Single file
python3 inference_konkanivani.py \
    --checkpoint checkpoints/best_model.pt \
    --vocab vocab.json \
    --audio path/to/audio.wav \
    --device mps

# Batch processing
python3 inference_konkanivani.py \
    --checkpoint checkpoints/best_model.pt \
    --vocab vocab.json \
    --audio path/to/audio_directory/ \
    --output transcripts.json \
    --device mps
```

## 📊 Expected Performance

With 26 hours of training data:
- **Word Error Rate (WER)**: 25-35%
- **Character Error Rate (CER)**: 15-25%
- **Inference Speed**: < 2x real-time (RTF < 2.0)

## 🔧 Configuration

Edit `config/model_config.yaml` to customize:
- Model dimensions (d_model, num_layers, num_heads)
- Training hyperparameters (learning_rate, batch_size)
- Data augmentation settings (SpecAugment)
- Loss weights (CTC vs Attention)

## 📝 Model Details

### Audio Processing
- **Sample Rate**: 16kHz
- **Features**: 80-dim mel-spectrogram
- **Augmentation**: SpecAugment (time + frequency masking)

### Text Processing
- **Tokenization**: Character-level (Devanagari)
- **Special Tokens**: `<pad>`, `<blank>`, `<sos>`, `<eos>`, `<unk>`
- **Vocabulary Size**: 200 characters

### Training
- **Optimizer**: AdamW (lr=0.0005, weight_decay=1e-6)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Gradient Clipping**: 5.0
- **Mixed Precision**: Supported

## 🎓 Citation

If you use KonkaniVani in your research, please cite:

```bibtex
@software{konkanivani2024,
  title={KonkaniVani: Custom ASR for Konkani Language},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/konkanivani}
}
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- Omnilingual ASR for initial transcription
- Hugging Face Transformers for inspiration
- PyTorch team for the framework
