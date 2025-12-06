# Konkani NLP Project

A comprehensive Natural Language Processing project for Konkani language sentiment analysis and ASR, featuring custom neural networks and modular architecture.

## 🎯 Project Overview

This project provides complete infrastructure for Konkani NLP:
- **Custom BiLSTM Model**: Built from scratch with attention mechanism
- **Transfer Learning**: Fine-tuned multilingual BERT models
- **ASR Support**: Audio processing and speech recognition (in development)
- **Modular Architecture**: Professional Python package structure
- **Production Ready**: Validated, tested, and packaged for deployment

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Stavin13/konkani-vaani-asr.git
cd konkani-vaani-asr/konkani

# Install the package
pip install -e .

# Or install with specific features
pip install -e ".[sentiment]"  # For sentiment analysis only
pip install -e ".[asr]"         # For ASR only
pip install -e ".[all]"         # For everything
```

## 🚀 Quick Start

### Training a Sentiment Model

```bash
python scripts/sentiment/train_custom.py
```

### Making Predictions

```bash
python scripts/sentiment/predict.py
```

Or use in your code:

```python
from konkani.inference import SentimentPredictor

predictor = SentimentPredictor("models/sentiment/custom_konkani_model")
result = predictor.predict("हें फोन खूब छान आसा")
print(result)
# {'label': 'positive', 'confidence': 0.92, ...}
```

## 📁 Project Structure

```
konkani/
├── konkani/              # Main package
│   ├── core/            # Tokenizer, Dataset, Metrics
│   ├── models/          # Model architectures
│   ├── training/        # Training infrastructure
│   ├── inference/       # Prediction classes
│   ├── data/            # Data processing
│   └── utils/           # Utilities
│
├── config/              # Configuration management
│   ├── paths.py         # Path management
│   ├── model_config.py  # Model hyperparameters
│   └── training_config.py  # Training settings
│
├── scripts/             # Executable scripts
│   ├── sentiment/       # Sentiment analysis scripts
│   ├── asr/             # ASR scripts
│   └── data/            # Data management scripts
│
├── data/                # Datasets
│   ├── raw/             # Original data
│   ├── processed/       # Processed data
│   └── cache/           # Temporary files
│
├── models/              # Trained models
│   ├── sentiment/       # Sentiment models
│   └── asr/             # ASR models
│
├── docs/                # Documentation
└── tests/               # Unit tests
```

## 📊 Dataset

- **Total Entries**: 47,922
- **Unique Texts**: 13,674 (Devanagari)
- **Labels**: Negative, Neutral, Positive (balanced)
- **Formats**: CSV, JSONL, JSON
- **Languages**: Devanagari + Romanized variants

## 🤖 Models

### Custom BiLSTM Model
- **Architecture**: BiLSTM with Attention
- **Parameters**: ~3-5M
- **Accuracy**: 82-88%
- **Speed**: Very Fast (10-20ms/sentence)

### Transfer Learning Model
- **Base**: DistilBERT Multilingual
- **Parameters**: 66M
- **Accuracy**: 85-92%
- **Speed**: Moderate (50-100ms/sentence)

## 🛠️ Development

### Package Structure

The codebase follows modern Python packaging best practices:

```python
# Import core components
from konkani.core import KonkaniTokenizer, KonkaniDataset
from konkani.models.sentiment.bilstm import CustomKonkaniSentimentModel
from konkani.training import SentimentTrainer
from konkani.inference import SentimentPredictor

# Use configuration
from config.paths import Paths
from config.model_config import BiLSTMConfig
from config.training_config import TrainingConfig
```

### Adding New Features

1. **New Model**: Add to `konkani/models/`
2. **New Trainer**: Add to `konkani/training/`
3. **New Script**: Add to `scripts/`
4. **Configuration**: Update `config/`

See [docs/architecture.md](docs/architecture.md) for detailed architecture overview.

## 📖 Documentation

- **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrading from old structure
- **[Architecture](docs/architecture.md)** - System design and components
- **[Training Guide](docs/README_TRAINING.md)** - Transfer learning setup
- **[Custom Model Guide](docs/README_CUSTOM_MODEL.md)** - BiLSTM architecture
- **[Model Comparison](docs/MODEL_COMPARISON.md)** - Choose the right model

## ✅ Features

- ✅ **Modular Design**: Clean separation of concerns
- ✅ **Configuration Management**: Centralized settings
- ✅ **Type Safety**: Dataclasses for configuration
- ✅ **Path Management**: No hardcoded paths
- ✅ **Proper Packaging**: Installable via pip
- ✅ **Documentation**: Comprehensive guides
- ✅ **Extensible**: Easy to add new models/features

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_tokenizer.py

# With coverage
pytest --cov=konkani tests/
```

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- GitHub: [@Stavin13](https://github.com/Stavin13)
- Repository: [konkani-vaani-asr](https://github.com/Stavin13/konkani-vaani-asr)

---

**Built with ❤️ for Konkani NLP**
