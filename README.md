# Konkani NLP Project

A comprehensive NLP project for Konkani language including ASR, NER, and Translation models.

## Project Structure

```
.
├── archives/           # Model checkpoints and zip files
├── checkpoints/        # Training checkpoints
├── config/            # Configuration files
├── data/              # Data files and datasets
├── docs/              # Technical documentation
├── documentation/     # Setup guides and tutorials
├── KonkaniRawSpeechCorpus/  # Raw speech corpus
├── logs/              # Training logs
├── models/            # Model implementations
├── notebooks/         # Jupyter notebooks for training
├── outputs/           # Model outputs and reports
├── scripts/           # Utility scripts
├── src/               # Source code
├── training_scripts/  # Python training scripts
└── utilities/         # Helper scripts and tools
```

## Quick Start

See the documentation folder for setup guides:
- `documentation/QUICK_START.md` - General quick start
- `documentation/ASR_QUICK_SETUP.md` - ASR setup
- `documentation/QUICK_START_NER.md` - NER setup
- `documentation/COLAB_SETUP_GUIDE.md` - Google Colab setup

## Requirements

```bash
pip install -r requirements.txt
```

## Training

Training scripts are located in:
- `training_scripts/` - Python training scripts
- `notebooks/` - Jupyter notebooks for interactive training

## Models

- ASR (Automatic Speech Recognition)
- NER (Named Entity Recognition)
- Translation (Konkani-English)
