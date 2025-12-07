# Architecture Overview

## Package Structure

The Konkani NLP package is organized into several key components:

```
konkani/
├── core/              # Core utilities
│   ├── tokenizer.py   # Text tokenization
│   ├── dataset.py     # PyTorch datasets
│   └── metrics.py     # Evaluation metrics
│
├── models/            # Model architectures
│   ├── sentiment/     # Sentiment models
│   │   └── bilstm.py  # BiLSTM with attention
│   └── asr/           # ASR models (future)
│
├── training/          # Training infrastructure
│   └── sentiment_trainer.py  # Sentiment trainer
│
├── inference/         # Inference/prediction
│   └── sentiment_predictor.py  # Sentiment predictor
│
├── data/              # Data processing
│   ├── preprocessing.py      # Text preprocessing
│   ├── audio_processing.py   # Audio processing
│   └── augmentation.py       # Data augmentation
│
└── utils/             # Utilities
    ├── io.py          # File I/O
    ├── visualization.py  # Plotting
    └── logging.py     # Logging setup
```

## Configuration

Centralized configuration in `config/`:

- **paths.py**: All file paths (data, models, outputs)
- **model_config.py**: Model hyperparameters (BiLSTM, Transformer, ASR)
- **training_config.py**: Training settings (batch size, learning rate, etc.)

## Data Flow

### Training Pipeline

```
Raw Data → Preprocessing → Tokenization → Dataset → DataLoader → Model → Training → Evaluation
```

1. **Data Loading**: Load CSV/JSON from `data/processed/sentiment/`
2. **Tokenization**: Build vocabulary and encode text
3. **Dataset Creation**: Create PyTorch datasets with padding
4. **Training**: Train with early stopping and checkpointing
5. **Evaluation**: Calculate metrics and generate visualizations

### Inference Pipeline

```
Text Input → Tokenization → Model → Softmax → Prediction
```

1. **Load Model**: Load trained model and tokenizer
2. **Encode**: Tokenize and encode input text
3. **Predict**: Forward pass through model
4. **Decode**: Convert logits to probabilities and labels

## Key Design Decisions

### 1. Modular Architecture

Each component is self-contained and can be imported independently:
```python
from konkani.core import KonkaniTokenizer
from konkani.models.sentiment.bilstm import CustomKonkaniSentimentModel
from konkani.training import SentimentTrainer
```

### 2. Configuration-Driven

Settings are defined in dataclasses for type safety and validation:
```python
from config.model_config import BiLSTMConfig
config = BiLSTMConfig(vocab_size=10000, embedding_dim=256)
```

### 3. Path Management

All paths are managed through `Paths` class:
```python
from config.paths import Paths
data_path = Paths.DATA_PROCESSED_SENTIMENT / "dataset.csv"
model_path = Paths.get_model_path("sentiment", "custom")
```

### 4. Separation of Concerns

- **Models**: Define architecture only
- **Trainers**: Handle training loop, optimization, checkpointing
- **Predictors**: Handle inference and post-processing
- **Utils**: Provide reusable helper functions

## Extension Points

### Adding a New Model

1. Create model file in `konkani/models/sentiment/new_model.py`
2. Implement `nn.Module` with `forward()` method
3. Add to `konkani/models/sentiment/__init__.py`
4. Create trainer if needed in `konkani/training/`

### Adding ASR Support

1. Implement models in `konkani/models/asr/`
2. Create ASR trainer in `konkani/training/asr_trainer.py`
3. Add audio processing in `konkani/data/audio_processing.py`
4. Create ASR scripts in `scripts/asr/`

### Adding New Features

1. **Data Processing**: Add to `konkani/data/`
2. **Utilities**: Add to `konkani/utils/`
3. **Configuration**: Add to `config/`
4. **Scripts**: Add to `scripts/`

## Testing Strategy

Tests should be organized in `tests/`:
```
tests/
├── test_tokenizer.py
├── test_models.py
├── test_data.py
└── test_training.py
```

Run tests with:
```bash
pytest tests/
```

## Deployment

The package can be installed and used in production:

```bash
# Install
pip install -e .

# Use in code
from konkani.inference import SentimentPredictor
predictor = SentimentPredictor("models/sentiment/custom_konkani_model")
result = predictor.predict("हें फोन खूब छान आसा")
```
