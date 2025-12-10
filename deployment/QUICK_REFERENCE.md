# KonkaniVani - Quick Reference Card

## 🚀 Getting Started (30 seconds)

```bash
cd deployment
./run.sh
```

Opens web app at `http://localhost:8501`

## 📝 Command Line

```bash
# Process audio
python pipeline.py --audio file.wav

# Process text
python pipeline.py --text "हांव घरा वचता"

# Specify device
python pipeline.py --text "..." --device mps
```

## 🐍 Python API

```python
from pipeline import KonkaniPipeline

# Initialize
pipeline = KonkaniPipeline(device='mps')

# Process audio
results = pipeline.process_audio('audio.wav')

# Process text
results = pipeline.process_text('हांव घरा वचता')
```

## 🎯 Individual Models

```python
# ASR only
from models import ASRModel
asr = ASRModel()
text = asr.transcribe('audio.wav')

# Translation only
from models import TranslationModel
translator = TranslationModel()
english = translator.konkani_to_english('हांव घरा वचता')

# Emotion only
from models import EmotionModel
emotion_model = EmotionModel()
emotion, conf, scores = emotion_model.predict('हांव खुश आसा')

# NER only
from models import NERModel
ner = NERModel()
entities = ner.predict('हांव Mumbai वचता')
```

## 📊 Output Format

```json
{
  "konkani_text": "हांव घरा वचता",
  "english_text": "I am going home",
  "emotion": {
    "label": "neutral",
    "confidence": 0.85,
    "all_scores": {...}
  },
  "entities": [
    ["Mumbai", "LOC", 1, 1]
  ]
}
```

## 🔧 Troubleshooting

```bash
# Verify setup
python test_pipeline.py

# Run demo
python demo.py

# Check imports
python -c "from models import *; print('OK')"
```

## 📁 Key Files

- `app.py` - Web interface
- `pipeline.py` - Main pipeline + CLI
- `models/*.py` - Model wrappers
- `README.md` - Quick start
- `USAGE_GUIDE.md` - Detailed guide

## 💡 Tips

- Use GPU for 5-10x speedup: `device='mps'` or `device='cuda'`
- Toggle analyses: `include_translation=False`
- Process batches for efficiency
- Cache pipeline for multiple requests

## 🎨 Web Interface

1. Click "Initialize Pipeline"
2. Upload audio OR enter text
3. Select analyses to run
4. Click "Process"
5. Download results as JSON

## 🆘 Help

- Check `USAGE_GUIDE.md` for examples
- Run `python test_pipeline.py` to diagnose
- See `DEPLOYMENT_SUMMARY.md` for details
