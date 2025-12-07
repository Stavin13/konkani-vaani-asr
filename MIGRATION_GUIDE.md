# Konkani NLP Restructuring - Migration Guide

## Overview

The Konkani NLP codebase has been restructured into a modern Python package with better organization and modularity.

## What Changed

### 1. New Package Structure

The project is now a proper Python package named `konkani-nlp`:
- `src/` → `konkani/` (main package)
- Added `config/` for centralized configuration
- Added `pyproject.toml` and `setup.py` for package installation

### 2. Modular Components

Code has been extracted into focused modules:
- **Core**: `konkani/core/` - Tokenizer, Dataset, Metrics
- **Models**: `konkani/models/sentiment/` - Model architectures
- **Training**: `konkani/training/` - Training infrastructure
- **Inference**: `konkani/inference/` - Prediction classes
- **Utils**: `konkani/utils/` - Visualization, I/O, Logging
- **Data**: `konkani/data/` - Preprocessing utilities

### 3. Configuration Management

All paths and settings are now centralized:
- `config/paths.py` - Path management
- `config/model_config.py` - Model hyperparameters
- `config/training_config.py` - Training settings

### 4. Reorganized Data

Data is now organized by type:
```
data/
├── raw/
│   ├── sentiment/  # GomEn datasets
│   └── asr/        # Parquet files
├── processed/
│   ├── sentiment/  # Processed sentiment data
│   └── asr/        # Processed ASR data
└── cache/          # Temporary files
```

### 5. Reorganized Models

Models are organized by type:
```
models/
├── sentiment/
│   └── custom_konkani_model/
└── asr/
```

## Migration Steps

### 1. Install the Package

```bash
cd /Users/stavinfernandes/Desktop/NLP/konkani
pip install -e .
```

This installs the package in editable mode, making all modules importable.

### 2. Update Import Statements

**Old imports:**
```python
from src.training.train_custom_model import KonkaniTokenizer
from src.training.train_custom_model import CustomKonkaniSentimentModel
```

**New imports:**
```python
from konkani.core import KonkaniTokenizer
from konkani.models.sentiment.bilstm import CustomKonkaniSentimentModel
```

### 3. Use New Scripts

**Training:**
```bash
# Old
python src/training/train_custom_model.py

# New
python scripts/sentiment/train_custom.py
```

**Prediction:**
```bash
# Old
python src/inference/test_custom_model.py

# New
python scripts/sentiment/predict.py
```

### 4. Use Configuration Classes

**Old:**
```python
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
# ... hardcoded values
```

**New:**
```python
from config.model_config import BiLSTMConfig
from config.training_config import TrainingConfig
from config.paths import Paths

model_config = BiLSTMConfig(vocab_size=10000, embedding_dim=256)
training_config = TrainingConfig(batch_size=32, num_epochs=50)
data_path = Paths.DATA_PROCESSED_SENTIMENT / "custom_konkani_sentiment_fixed.csv"
```

## Benefits

1. **Better Organization**: Clear separation of concerns
2. **Reusability**: Components can be imported and reused
3. **Maintainability**: Easier to find and update code
4. **Professionalism**: Follows Python packaging best practices
5. **Flexibility**: Easy to add new models or features
6. **Configuration**: Centralized settings management

## Backward Compatibility

The old scripts in `src/` are still present but should be considered deprecated. They will continue to work but won't receive updates.

## Next Steps

1. Test the new training script
2. Verify predictions work correctly
3. Update any external scripts or notebooks
4. Consider removing old `src/` directory after verification

## Questions?

If you encounter any issues with the migration, check:
1. Package is installed: `pip list | grep konkani`
2. Imports work: `python -c "import konkani; print(konkani.__version__)"`
3. Paths are correct: Check `config/paths.py`
