# KonkaniVani Pipeline - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     KONKANIVANI PIPELINE                        │
│                   Complete NLP System for Konkani                │
└─────────────────────────────────────────────────────────────────┘

                              INPUT
                                │
                    ┌───────────┴───────────┐
                    │                       │
                 AUDIO                    TEXT
                  .wav                  Konkani
                  .mp3                Devanagari
                  .flac
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    STREAMLIT APP      │
                    │   or CLI or Python    │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   PIPELINE MANAGER    │
                    │   (pipeline.py)       │
                    └───────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │    ASR    │   │TRANSLATION│   │  EMOTION  │
        │  Model    │   │   Model   │   │   Model   │
        └───────────┘   └───────────┘   └───────────┘
                │               │               │
                └───────────────┼───────────────┘
                                │
                                ▼
                        ┌───────────┐
                        │    NER    │
                        │   Model   │
                        └───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   STRUCTURED OUTPUT   │
                    │        (JSON)         │
                    └───────────────────────┘
```

## Component Details

### 1. Input Layer

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio Input                    Text Input                  │
│  ├─ WAV (16kHz mono)           ├─ Konkani (Devanagari)    │
│  ├─ MP3                        ├─ UTF-8 encoded            │
│  ├─ FLAC                       └─ Plain text               │
│  └─ OGG                                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Interface Layer

```
┌─────────────────────────────────────────────────────────────┐
│                     INTERFACE LAYER                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Streamlit  │  │     CLI     │  │  Python API │       │
│  │   Web App   │  │    Tool     │  │   Import    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│       │                  │                  │              │
│       └──────────────────┼──────────────────┘              │
│                          │                                 │
│                    app.py / pipeline.py                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Pipeline Layer

```
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE LAYER                          │
│                    (pipeline.py)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  class KonkaniPipeline:                                     │
│                                                             │
│    ┌─────────────────────────────────────────────┐        │
│    │  process_audio(audio_path)                  │        │
│    │    1. Load audio file                       │        │
│    │    2. Call ASR model                        │        │
│    │    3. Get Konkani text                      │        │
│    │    4. Continue to text processing...        │        │
│    └─────────────────────────────────────────────┘        │
│                                                             │
│    ┌─────────────────────────────────────────────┐        │
│    │  process_text(konkani_text)                 │        │
│    │    1. Translate to English                  │        │
│    │    2. Detect emotion                        │        │
│    │    3. Extract entities                      │        │
│    │    4. Return structured results             │        │
│    └─────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Model Layer

```
┌─────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                           │
│                    (models/*.py)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │  ASR Model (asr_model.py)                    │         │
│  │  ├─ Architecture: Conformer + Transformer    │         │
│  │  ├─ Input: Audio (mel-spectrogram)           │         │
│  │  ├─ Output: Konkani text                     │         │
│  │  └─ Checkpoint: best_model.pt (50 epochs)    │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │  Translation Model (translation_model.py)    │         │
│  │  ├─ Architecture: NLLB-200 (600M)            │         │
│  │  ├─ Input: Text (Konkani or English)         │         │
│  │  ├─ Output: Translated text                  │         │
│  │  └─ Checkpoint: nllb_finetuned or base       │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │  Emotion Model (emotion_model.py)            │         │
│  │  ├─ Architecture: BiLSTM + Attention         │         │
│  │  ├─ Input: Konkani text                      │         │
│  │  ├─ Output: Emotion + confidence             │         │
│  │  └─ Classes: 7 emotions                      │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
│  ┌──────────────────────────────────────────────┐         │
│  │  NER Model (ner_model.py)                    │         │
│  │  ├─ Architecture: BiLSTM-CRF                 │         │
│  │  ├─ Input: Konkani text                      │         │
│  │  ├─ Output: Named entities                   │         │
│  │  └─ Types: PER, ORG, LOC, MISC               │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. Output Layer

```
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  {                                                          │
│    "konkani_text": "हांव घरा वचता",                       │
│    "english_text": "I am going home",                      │
│    "emotion": {                                             │
│      "label": "neutral",                                    │
│      "confidence": 0.85,                                    │
│      "all_scores": {                                        │
│        "joy": 0.05,                                         │
│        "sadness": 0.03,                                     │
│        "anger": 0.02,                                       │
│        "fear": 0.01,                                        │
│        "surprise": 0.02,                                    │
│        "disgust": 0.02,                                     │
│        "neutral": 0.85                                      │
│      }                                                      │
│    },                                                       │
│    "entities": [                                            │
│      ["Mumbai", "LOC", 1, 1]                               │
│    ]                                                        │
│  }                                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Audio Processing Flow

```
Audio File (.wav)
    │
    ├─ Load with torchaudio
    │
    ├─ Resample to 16kHz
    │
    ├─ Convert to mono
    │
    ├─ Extract mel-spectrogram (80 dims)
    │
    ▼
ASR Model (Conformer + Transformer)
    │
    ├─ Encoder: Audio → Hidden states
    │
    ├─ CTC Head: Alignment-free decoding
    │
    ├─ Decoder: Attention-based refinement
    │
    ▼
Konkani Text (Devanagari)
    │
    └─ Continue to text processing...
```

### Text Processing Flow

```
Konkani Text
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
Translation      Emotion           NER              (Original)
    │                  │                  │                  │
    ├─ Tokenize        ├─ Char-level     ├─ Word tokens     │
    │                  │   encoding       │                  │
    ├─ NLLB            ├─ BiLSTM          ├─ Char features   │
    │   Encoder        │                  │                  │
    │                  ├─ Attention       ├─ BiLSTM          │
    ├─ NLLB            │   weights        │                  │
    │   Decoder        │                  ├─ CRF             │
    │                  ├─ Classifier      │   decoding       │
    ▼                  ▼                  ▼                  │
English Text     Emotion Label    Entity List              │
    │                  │                  │                  │
    └──────────────────┴──────────────────┴──────────────────┘
                            │
                            ▼
                    Structured JSON Output
```

## Model Architectures

### ASR Model (KonkaniVani)

```
Input: Mel-spectrogram (batch, time, 80)
    │
    ▼
┌─────────────────────────────────┐
│  Conformer Encoder (12 layers)  │
│  ├─ Feed-forward                │
│  ├─ Multi-head attention        │
│  ├─ Convolution                 │
│  └─ Feed-forward                │
└─────────────────────────────────┘
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
CTC Head    Transformer      Memory
            Decoder          for Decoder
            (6 layers)
    │              │
    └──────┬───────┘
           │
           ▼
    Konkani Text
```

### Translation Model (NLLB)

```
Input: Konkani Text
    │
    ▼
┌─────────────────────────────────┐
│  Tokenizer (SentencePiece)      │
│  ├─ Language code: kok_Deva     │
│  └─ Subword tokens               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Encoder (Transformer)           │
│  ├─ 24 layers                    │
│  ├─ 16 attention heads           │
│  └─ 1024 hidden dim              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Decoder (Transformer)           │
│  ├─ Language code: eng_Latn      │
│  ├─ Beam search (beam=5)         │
│  └─ Auto-regressive generation   │
└─────────────────────────────────┘
    │
    ▼
English Text
```

### Emotion Model

```
Input: Konkani Text
    │
    ▼
┌─────────────────────────────────┐
│  Character Embedding             │
│  ├─ Vocab size: ~5000            │
│  └─ Embedding dim: 128           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  BiLSTM (2 layers)               │
│  ├─ Hidden dim: 256              │
│  ├─ Bidirectional                │
│  └─ Dropout: 0.3                 │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Attention Layer                 │
│  ├─ Compute attention weights    │
│  └─ Weighted sum of LSTM outputs │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Classifier                      │
│  ├─ FC layer (256 → 7)           │
│  └─ Softmax                      │
└─────────────────────────────────┘
    │
    ▼
Emotion (7 classes)
```

### NER Model

```
Input: Konkani Text
    │
    ├──────────────┬──────────────┐
    │              │              │
    ▼              ▼              ▼
Word Embedding  Char Embedding  Position
    │              │
    │              ├─ CNN
    │              │
    │              ├─ Max pooling
    │              │
    └──────┬───────┘
           │
           ▼
┌─────────────────────────────────┐
│  BiLSTM (2 layers)               │
│  ├─ Hidden dim: 256              │
│  └─ Dropout: 0.3                 │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Linear Projection               │
│  └─ 256 → num_tags               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  CRF Layer                       │
│  ├─ Transition scores            │
│  └─ Viterbi decoding             │
└─────────────────────────────────┘
    │
    ▼
Entity Tags (BIO format)
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT OPTIONS                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Local Development                                          │
│  ├─ streamlit run app.py                                   │
│  └─ http://localhost:8501                                  │
│                                                             │
│  Command Line                                               │
│  ├─ python pipeline.py --audio file.wav                    │
│  └─ python pipeline.py --text "..."                        │
│                                                             │
│  Python Integration                                         │
│  ├─ from pipeline import KonkaniPipeline                   │
│  └─ pipeline = KonkaniPipeline()                           │
│                                                             │
│  Future: Docker                                             │
│  ├─ docker build -t konkanivani .                          │
│  └─ docker run -p 8501:8501 konkanivani                    │
│                                                             │
│  Future: Cloud                                              │
│  ├─ Streamlit Cloud                                        │
│  ├─ Heroku                                                 │
│  └─ AWS/GCP/Azure                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

```
┌─────────────────────────────────────────────────────────────┐
│                    PERFORMANCE METRICS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model          Size      Speed (Mac M1)    Accuracy       │
│  ─────────────────────────────────────────────────────     │
│  ASR            ~50MB     2-3s per 10s      High           │
│  Translation    2.4GB     0.5s per sent     High           │
│  Emotion        ~10MB     0.1s per text     Good           │
│  NER            ~20MB     0.2s per text     Good           │
│                                                             │
│  Total Pipeline: ~3-4s per audio file (10s)                │
│                  ~1s per text input                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Core Framework                                             │
│  ├─ PyTorch 2.0+                                           │
│  └─ Python 3.8+                                            │
│                                                             │
│  Models                                                     │
│  ├─ Transformers (Hugging Face)                            │
│  ├─ TorchAudio                                             │
│  └─ PyTorch-CRF                                            │
│                                                             │
│  Interface                                                  │
│  ├─ Streamlit (Web UI)                                     │
│  └─ Argparse (CLI)                                         │
│                                                             │
│  Utilities                                                  │
│  ├─ NumPy                                                  │
│  └─ JSON                                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
