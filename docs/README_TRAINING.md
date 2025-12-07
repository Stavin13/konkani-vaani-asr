# Konkani Sentiment Analysis Model Training Guide

This guide will help you train a custom sentiment analysis model for Konkani language using your custom dataset.

## 📋 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- At least 4GB RAM (8GB+ recommended)
- GPU optional but recommended for faster training

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install required packages
pip install -r requirements_training.txt
```

### 2. Train the Model

```bash
# Basic training (uses default settings)
python train_konkani_model.py
```

The training script will:
- Load your custom Konkani dataset (`custom_konkani_sentiment.csv`)
- Split it into train/validation/test sets (70%/15%/15%)
- Fine-tune a multilingual BERT model
- Save the trained model to `./konkani_sentiment_model/`
- Generate evaluation metrics and confusion matrix

### 3. Test the Model

```bash
# Test with sample sentences
python test_model.py
```

This will:
- Load your trained model
- Test it with predefined Konkani sentences
- Enter interactive mode for custom testing

## 📊 Training Configuration

### Default Settings

```python
MODEL_NAME = "distilbert-base-multilingual-cased"  # Fast and efficient
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
```

### Alternative Models

You can modify `train_konkani_model.py` to use different base models:

```python
# Option 1: DistilBERT (Default - Fast, 66M parameters)
MODEL_NAME = "distilbert-base-multilingual-cased"

# Option 2: BERT (More accurate, 110M parameters)
MODEL_NAME = "bert-base-multilingual-cased"

# Option 3: XLM-RoBERTa (Best multilingual, 270M parameters)
MODEL_NAME = "xlm-roberta-base"

# Option 4: MuRIL (Optimized for Indian languages, 237M parameters)
MODEL_NAME = "google/muril-base-cased"
```

## 📈 Training Process

### What Happens During Training

1. **Data Loading**: Loads your 47,922 entry dataset
2. **Preprocessing**: Tokenizes text and encodes labels
3. **Training**: Fine-tunes the model for 5 epochs
4. **Validation**: Evaluates on validation set every 500 steps
5. **Early Stopping**: Stops if no improvement for 3 evaluations
6. **Saving**: Saves best model based on F1 score

### Expected Training Time

- **CPU**: 2-4 hours
- **GPU (CUDA)**: 30-60 minutes
- **Apple Silicon (MPS)**: 45-90 minutes

### Output Files

After training, you'll find:

```
konkani_sentiment_model/
├── config.json                 # Model configuration
├── pytorch_model.bin           # Trained weights
├── tokenizer_config.json       # Tokenizer settings
├── vocab.txt                   # Vocabulary
├── README.md                   # Model card
├── confusion_matrix.png        # Evaluation visualization
├── test_results.json           # Test metrics
└── logs/                       # Training logs
```

## 🎯 Expected Performance

Based on your dataset size and quality, you should expect:

- **Accuracy**: 85-92%
- **F1 Score**: 0.83-0.90
- **Precision**: 0.82-0.89
- **Recall**: 0.84-0.91

## 🔧 Customization

### Adjust Training Parameters

Edit `train_konkani_model.py`:

```python
# For faster training (lower accuracy)
NUM_EPOCHS = 3
BATCH_SIZE = 32

# For better accuracy (slower training)
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
```

### Use Different Dataset

```python
# In train_konkani_model.py, modify:
DATA_PATH = "your_dataset.csv"  # or .jsonl
DATA_FORMAT = "csv"  # or "jsonl"
```

### Change Train/Val/Test Split

```python
dataset_dict = trainer.prepare_datasets(
    df,
    test_size=0.20,   # 20% for testing
    val_size=0.15,    # 15% for validation
    random_state=42
)
```

## 📱 Using Your Trained Model

### Python API

```python
from transformers import pipeline

# Load your model
classifier = pipeline(
    'sentiment-analysis',
    model='./konkani_sentiment_model'
)

# Predict
result = classifier("हें फोन खूब छान आसा")
print(result)
# [{'label': 'positive', 'score': 0.95}]
```

### Batch Prediction

```python
sentences = [
    "हें फोन खूब छान आसा",
    "ही सेवा वायट आसा",
    "हें पुस्तक साधारण आसा"
]

results = classifier(sentences)
for sent, res in zip(sentences, results):
    print(f"{sent} → {res['label']} ({res['score']:.2%})")
```

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
BATCH_SIZE = 8  # or even 4

# Reduce max sequence length
MAX_LENGTH = 64
```

### Slow Training

```python
# Use DistilBERT instead of BERT
MODEL_NAME = "distilbert-base-multilingual-cased"

# Enable mixed precision (if GPU available)
# Already enabled in training script
```

### Poor Accuracy

```python
# Increase training epochs
NUM_EPOCHS = 10

# Use a larger model
MODEL_NAME = "xlm-roberta-base"

# Reduce learning rate
LEARNING_RATE = 1e-5
```

## 📊 Monitoring Training

Watch the training progress:

```bash
# Training will show:
# - Loss (should decrease)
# - Accuracy (should increase)
# - F1 Score (should increase)
# - Validation metrics every 500 steps
```

## 🚀 Next Steps

After training:

1. **Evaluate**: Check `test_results.json` and `confusion_matrix.png`
2. **Test**: Run `python test_model.py` with your own sentences
3. **Deploy**: Use the model in your application
4. **Share**: Upload to HuggingFace Hub (optional)

## 📚 Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Model Hub](https://huggingface.co/models)

## 💡 Tips

1. **Start Small**: Test with a subset first to ensure everything works
2. **Monitor GPU**: Use `nvidia-smi` or `watch -n 1 nvidia-smi`
3. **Save Checkpoints**: The script saves every 500 steps
4. **Experiment**: Try different models and hyperparameters
5. **Validate**: Always check performance on real-world examples

## 🎓 Understanding the Results

### Confusion Matrix

Shows how well the model distinguishes between sentiments:
- Diagonal = Correct predictions
- Off-diagonal = Misclassifications

### Metrics Explained

- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many were predicted
- **F1 Score**: Harmonic mean of precision and recall

---

**Happy Training! 🚀**

For questions or issues, check the training logs in `konkani_sentiment_model/logs/`
