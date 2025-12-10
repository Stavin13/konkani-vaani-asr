# KonkaniVani Pipeline - Usage Guide

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Web Interface](#web-interface)
4. [Command Line](#command-line)
5. [Python API](#python-api)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- Optional: GPU (Mac M1/M2 or NVIDIA)

### Step 1: Install Dependencies

```bash
cd deployment
pip install -r requirements.txt
```

### Step 2: Verify Setup

```bash
python test_pipeline.py
```

This will check:
- All required packages are installed
- Model checkpoints exist
- Available compute devices

## Quick Start

### Option 1: Quick Start Script (Recommended)

```bash
./run.sh
```

This will:
1. Create a virtual environment (if needed)
2. Install dependencies
3. Launch the Streamlit app

### Option 2: Manual Start

```bash
streamlit run app.py
```

## Web Interface

### 1. Initialize Pipeline

1. Open the app in your browser (usually `http://localhost:8501`)
2. In the sidebar, click **"🚀 Initialize Pipeline"**
3. Wait for models to load (~30 seconds)

### 2. Process Audio

**Tab: 🎤 Audio Input**

1. Click "Browse files" to upload audio
2. Supported formats: WAV, MP3, FLAC, OGG
3. Preview the audio
4. Click **"🎯 Process Audio"**
5. View results:
   - Konkani transcription
   - English translation
   - Emotion analysis
   - Named entities

### 3. Process Text

**Tab: ✍️ Text Input**

1. Type or paste Konkani text (Devanagari script)
2. Click **"🎯 Process Text"**
3. View results

### 4. Configure Processing

In the sidebar, toggle:
- ☑️ **Translation** - Translate to English
- ☑️ **Emotion Detection** - Detect emotion
- ☑️ **Named Entity Recognition** - Extract entities

### 5. Download Results

Click **"📥 Download Results (JSON)"** to save results as JSON file.

## Command Line

### Basic Usage

```bash
# Process audio file
python pipeline.py --audio path/to/audio.wav

# Process text
python pipeline.py --text "हांव घरा वचता"
```

### Advanced Options

```bash
# Specify device
python pipeline.py --text "तूं कसो आसा?" --device mps

# Demo mode (no input)
python pipeline.py
```

## Python API

### Complete Pipeline

```python
from pipeline import KonkaniPipeline

# Initialize
pipeline = KonkaniPipeline(device='mps')  # or 'cuda', 'cpu'

# Process audio
results = pipeline.process_audio(
    'audio.wav',
    include_translation=True,
    include_emotion=True,
    include_ner=True
)

# Process text
results = pipeline.process_text(
    'हांव घरा वचता',
    include_translation=True,
    include_emotion=True,
    include_ner=True
)

print(results)
```

### Individual Models

#### ASR Only

```python
from models import ASRModel

asr = ASRModel(device='mps')
transcription = asr.transcribe('audio.wav')
print(f"Konkani: {transcription}")
```

#### Translation Only

```python
from models import TranslationModel

translator = TranslationModel(device='mps')

# Konkani to English
english = translator.konkani_to_english('हांव घरा वचता')
print(f"English: {english}")

# English to Konkani
konkani = translator.english_to_konkani('I am going home')
print(f"Konkani: {konkani}")
```

#### Emotion Detection Only

```python
from models import EmotionModel

emotion_model = EmotionModel(device='mps')
emotion, confidence, all_scores = emotion_model.predict('हांव खुश आसा')

print(f"Emotion: {emotion}")
print(f"Confidence: {confidence:.2%}")
print(f"All scores: {all_scores}")
```

#### NER Only

```python
from models import NERModel

ner = NERModel(device='mps')
entities = ner.predict('हांव Mumbai वचता')

for entity_text, entity_type, start, end in entities:
    print(f"{entity_type}: {entity_text}")
```

## Examples

### Example 1: Audio Transcription + Translation

```python
from pipeline import KonkaniPipeline

pipeline = KonkaniPipeline()

results = pipeline.process_audio(
    'konkani_speech.wav',
    include_translation=True,
    include_emotion=False,
    include_ner=False
)

print(f"Konkani: {results['konkani_text']}")
print(f"English: {results['english_text']}")
```

### Example 2: Emotion Analysis

```python
from pipeline import KonkaniPipeline

pipeline = KonkaniPipeline()

texts = [
    'हांव खुश आसा',
    'हांव दुःखी आसा',
    'हांव रागीत आसा'
]

for text in texts:
    results = pipeline.process_text(
        text,
        include_translation=False,
        include_emotion=True,
        include_ner=False
    )
    
    emotion = results['emotion']
    print(f"{text} → {emotion['label']} ({emotion['confidence']:.1%})")
```

### Example 3: Entity Extraction

```python
from pipeline import KonkaniPipeline

pipeline = KonkaniPipeline()

text = 'हांव Mumbai सावन Goa वचता'

results = pipeline.process_text(
    text,
    include_translation=True,
    include_emotion=False,
    include_ner=True
)

print(f"Text: {text}")
print(f"Translation: {results['english_text']}")
print("\nEntities:")
for entity_text, entity_type, start, end in results['entities']:
    print(f"  {entity_type}: {entity_text}")
```

### Example 4: Batch Processing

```python
from models import TranslationModel

translator = TranslationModel()

konkani_texts = [
    'घर',
    'पाणी',
    'खाणे',
    'हांव घरा वचता'
]

for text in konkani_texts:
    english = translator.konkani_to_english(text)
    print(f"{text} → {english}")
```

## Troubleshooting

### Issue: Models not loading

**Solution:**
1. Run `python test_pipeline.py` to check setup
2. Verify checkpoint paths in model files
3. Check available RAM (need 8GB+)

### Issue: NLLB download fails

**Solution:**
1. Check internet connection
2. Model downloads ~2.4GB on first run
3. Subsequent runs use cached model

### Issue: Out of memory

**Solution:**
1. Use CPU instead of GPU: `device='cpu'`
2. Close other applications
3. Process shorter audio files
4. Disable some analyses

### Issue: Slow processing

**Solution:**
1. Use GPU if available: `device='mps'` or `device='cuda'`
2. Process text instead of audio (faster)
3. Disable unused analyses

### Issue: Audio file not supported

**Solution:**
1. Convert to WAV format
2. Use 16kHz sample rate
3. Use mono audio

### Issue: Incorrect transcriptions

**Solution:**
1. Use clear audio with minimal background noise
2. Ensure audio is in Konkani
3. Check audio quality (16kHz, 16-bit recommended)

### Issue: Translation quality

**Solution:**
1. If finetuned model exists, ensure it's being loaded
2. Base NLLB may have limited Konkani support
3. Consider finetuning on more data

## Performance Tips

1. **Use GPU**: 5-10x faster than CPU
2. **Batch processing**: Process multiple texts together
3. **Disable unused features**: Skip analyses you don't need
4. **Cache pipeline**: Reuse initialized pipeline for multiple requests
5. **Optimize audio**: Use 16kHz mono WAV files

## Support

For issues or questions:
1. Check this guide
2. Run `python test_pipeline.py`
3. Check main project README
4. Review model-specific documentation

## Next Steps

- Fine-tune NLLB on your Konkani data
- Train emotion model on more data
- Expand NER entity types
- Add more languages
- Deploy as web service
