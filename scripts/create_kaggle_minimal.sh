#!/bin/bash

# Create minimal package for Kaggle (much smaller!)
echo "📦 Creating minimal Kaggle package..."

# Create temp directory
mkdir -p kaggle_minimal

# Copy only essential files
echo "Copying essential files..."

# 1. Training script
mkdir -p kaggle_minimal/training_scripts
cp training_scripts/train_konkanivani_asr.py kaggle_minimal/training_scripts/

# 2. Model code
mkdir -p kaggle_minimal/models
cp models/konkanivani_asr.py kaggle_minimal/models/

# 3. Data processing code
mkdir -p kaggle_minimal/data/audio_processing
cp data/audio_processing/*.py kaggle_minimal/data/audio_processing/

# 4. Vocab and manifests (small files)
cp data/vocab.json kaggle_minimal/data/
mkdir -p kaggle_minimal/data/konkani-asr-v0/splits/manifests
cp data/konkani-asr-v0/splits/manifests/*.json kaggle_minimal/data/konkani-asr-v0/splits/manifests/

# 5. Checkpoint (large but necessary)
mkdir -p kaggle_minimal/archives
cp archives/checkpoint_epoch_15.pt kaggle_minimal/archives/

# 6. Audio files (the big one - but we need them)
echo "Copying audio files (this takes a moment)..."
mkdir -p kaggle_minimal/data/konkani-asr-v0/audio
cp -r data/konkani-asr-v0/audio/* kaggle_minimal/data/konkani-asr-v0/audio/

# Create zip
echo "Creating zip file..."
cd kaggle_minimal
zip -r ../kaggle_konkani_minimal.zip . -q
cd ..

# Show size
echo ""
echo "✅ Created: kaggle_konkani_minimal.zip"
ls -lh kaggle_konkani_minimal.zip

echo ""
echo "📊 Size comparison:"
echo "   Full project: ~14 GB"
echo "   Minimal package: $(du -sh kaggle_konkani_minimal.zip | cut -f1)"
echo ""
echo "📤 Upload this file to Kaggle!"
