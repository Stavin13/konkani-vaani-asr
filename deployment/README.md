# KonkaniVani - Complete NLP Pipeline

Complete Konkani language processing system with 4 integrated models:

1. **ASR (Automatic Speech Recognition)** - Transcribe Konkani audio to text
2. **Translation** - Translate between Konkani and English using NLLB
3. **Emotion Detection** - Detect emotions in Konkani text
4. **NER (Named Entity Recognition)** - Extract named entities

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Checkpoints

Make sure these checkpoint files exist:
- `../kaggle_best_model/checkpoints/best_model.pt` (ASR)
- `../checkpoints/ner/best_ner_model.pt` (NER)
- `../checkpoints/emotion_model/emotion_model_mac.pt` (Emotion)
- `../checkpoints/nllb_finetuned/final/` (Translation - optional, will use base NLLB if not found)

### 3. Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Streamlit Web Interface

1. Click "Initialize Pipeline" in the sidebar
2. Choose input mode:
   - **Audio Input**: Upload a Konkani audio file
   - **Text Input**: Type or paste Konkani text
3. Select which analyses to run (translation, emotion, NER)
4. Click "Process" to get results
5. Download results as JSON

### Command Line

```bash
# Process audio file
python pipeline.py --audio path/to/audio.wav

# Process text
python pipeline.py --text "हांव घरा वचता"

# Specify device
python pipeline.py --text "हांव घरा वचता" --device mps
```

## 🏗️ Architecture

```
deployment/
├── models/
│   ├── __init__.py
│   ├── asr_model.py          # ASR wrapper
│   ├── translation_model.py  # NLLB wrapper
│   ├── emotion_model.py      # Emotion detection wrapper
│   └── ner_model.py          # NER wrapper
├── pipeline.py               # Main pipeline orchestrator
├── app.py                    # Streamlit frontend
├── requirements.txt
└── README.md
```

## 🎯 Features

### ASR Model
- Conformer encoder + Transformer decoder
- CTC + Attention hybrid training
- 50 epochs trained on Konkani speech corpus

### Translation Model
- NLLB-200 (600M distilled)
- Finetuned on Konkani-English pairs
- Bidirectional translation support

### Emotion Model
- BiLSTM + Attention architecture
- 7 emotion classes: joy, sadness, anger, fear, surprise, disgust, neutral
- Character-level processing

### NER Model
- BiLSTM-CRF architecture
- Detects: Person, Organization, Location, Miscellaneous
- Character + word embeddings

## 🔧 Configuration

Edit `pipeline.py` to customize:
- Model checkpoint paths
- Device selection (CPU/GPU/MPS)
- Processing options

## 📊 Output Format

```json
{
  "konkani_text": "हांव घरा वचता",
  "english_text": "I am going home",
  "emotion": {
    "label": "neutral",
    "confidence": 0.85,
    "all_scores": {
      "joy": 0.05,
      "sadness": 0.03,
      "neutral": 0.85,
      ...
    }
  },
  "entities": [
    ["Mumbai", "LOC", 5, 5]
  ]
}
```

## 🐛 Troubleshooting

### Model Loading Errors
- Verify checkpoint paths in model wrappers
- Check that all checkpoint files exist
- Ensure sufficient RAM/VRAM

### NLLB Download Issues
- First run downloads ~2.4GB model
- Requires internet connection initially
- Cached locally after first download

### Device Errors
- Mac: Use `device='mps'` for GPU
- NVIDIA: Use `device='cuda'`
- Fallback: Use `device='cpu'`

## 📝 License

See parent project LICENSE file.

## 🤝 Contributing

This is part of the larger KonkaniVani project. See main README for contribution guidelines.
