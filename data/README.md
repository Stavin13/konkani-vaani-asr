# Data Directory

This directory contains all datasets used in the Konkani NLP project.

## Structure

```
data/
├── raw/          # Original, unmodified datasets
├── generated/    # Generated sentiment datasets
└── processed/    # Production-ready datasets
```

## Raw Data (`raw/`)

Original datasets from external sources.

### GomEn Dataset
- **Source**: Konkani-English parallel corpus
- **Files**:
  - `GomEn_ann1_train.json` - Training set
  - `GomEn_ann1_valid.json` - Validation set
  - `GomEn_ann1_test.json` - Test set
- **Purpose**: Reference data for Konkani language

## Generated Data (`generated/`)

Datasets created by our generation scripts.

### Sentiment Datasets
- `konkani_sentiment.csv` - Basic sentiment dataset
- `konkani_sentiment.jsonl` - JSONL format
- `konkani_sentiment_words.json` - Word-level sentiments
- `konkani_sentiment_sentences.json` - Sentence-level sentiments
- `konkani_large_dataset.json` - Large combined dataset
- `konkani_sentiment_10000.csv` - 10K sample dataset

### Custom Datasets
- `custom_konkani_dataset.json` - Full custom dataset (47,922 entries)
- `custom_konkani_dict.json` - Dictionary format
- `custom_konkani_sentiment.csv` - CSV format for ML
- `custom_konkani_sentiment.jsonl` - HuggingFace format

**Generation Scripts**: See `scripts/data_generation/`

## Processed Data (`processed/`)

Production-ready datasets with validation and quality checks.

### Production Dataset
- **File**: `custom_konkani_sentiment_fixed.csv`
- **Entries**: 47,922
- **Unique Texts**: 13,674 (Devanagari)
- **Labels**: Negative (33.4%), Neutral (33.3%), Positive (33.2%)
- **Validation**: ✅ No data leakage, UTF-8 validated
- **Splits**: Train (70%), Val (15%), Test (15%)

**Use this dataset for training production models.**

## Data Format

### CSV Format
```csv
id,text,devanagari,label,type,source,split
1,hen phone chan,हें फोन छान,positive,sentence,custom,train
```

### JSONL Format
```json
{"text": "hen phone chan", "label": "positive", "devanagari": "हें फोन छान"}
```

## Usage

### Load Production Dataset
```python
import pandas as pd

# Load processed dataset
df = pd.read_csv('data/processed/custom_konkani_sentiment_fixed.csv')

# Filter by split
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'validation']
test_df = df[df['split'] == 'test']
```

### Load Generated Dataset
```python
# Load custom dataset
df = pd.read_csv('data/generated/custom_konkani_sentiment.csv')
```

## Data Statistics

See `outputs/reports/custom_dataset_stats.json` for detailed statistics.

## Data Quality

- ✅ UTF-8 encoding validated
- ✅ NFC Unicode normalization
- ✅ No mojibake detected
- ✅ Balanced label distribution
- ✅ No data leakage between splits

## Regenerating Data

To regenerate datasets, run the generation scripts:

```bash
# Basic dataset
python scripts/data_generation/generate_sentiment_data.py

# Expanded dataset
python scripts/data_generation/generate_expanded_data.py

# Custom dataset (recommended)
python scripts/data_generation/generate_custom_dataset.py
```

---

**Always use `data/processed/custom_konkani_sentiment_fixed.csv` for production training.**
