#!/bin/bash

# Script to create a complete Kaggle dataset with ALL audio files from SSD
# This will help you upload the full 2,549+ training samples

echo "=========================================="
echo "Creating Complete Kaggle Dataset Package"
echo "=========================================="

# Audio files are on SSD
AUDIO_DIR="/Volumes/data&proj/konkani/data/konkani-asr-v0/data/processed_segments_diarized/audio_segments"

if [ ! -d "$AUDIO_DIR" ]; then
    echo ""
    echo "❌ Audio directory not found: $AUDIO_DIR"
    echo ""
    echo "Please make sure your SSD is mounted at /Volumes/data&proj/"
    echo ""
    exit 1
fi

# Count audio files
AUDIO_COUNT=$(find "$AUDIO_DIR" -name "*.wav" -type f | grep -v "/\._" | wc -l | tr -d ' ')

echo ""
echo "Found $AUDIO_COUNT audio files"
echo ""

if [ "$AUDIO_COUNT" -lt 2500 ]; then
    echo "⚠️  Warning: Expected ~2,549 audio files, but found only $AUDIO_COUNT"
    echo "   Some audio files may be missing!"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create package directory
PACKAGE_DIR="kaggle_complete_dataset"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

echo "Creating package structure..."

# Copy essential files
mkdir -p "$PACKAGE_DIR/data/konkani-asr-v0/audio"
mkdir -p "$PACKAGE_DIR/data/konkani-asr-v0/splits/manifests"
mkdir -p "$PACKAGE_DIR/models"
mkdir -p "$PACKAGE_DIR/data/audio_processing"
mkdir -p "$PACKAGE_DIR/training_scripts"
mkdir -p "$PACKAGE_DIR/checkpoints"

# Copy audio files (this is the important part!)
echo "Copying audio files... (this may take a while)"
cp "$AUDIO_DIR"/*.wav "$PACKAGE_DIR/data/konkani-asr-v0/audio/" 2>/dev/null || true

# Copy manifests from SSD
cp "/Volumes/data&proj/konkani/data/konkani-asr-v0/splits/manifests"/*.json "$PACKAGE_DIR/data/konkani-asr-v0/splits/manifests/"

# Copy vocab
cp data/vocab.json "$PACKAGE_DIR/data/" 2>/dev/null || cp "/Volumes/data&proj/konkani/data/vocab.json" "$PACKAGE_DIR/data/"

# Copy model files
cp models/konkanivani_asr.py "$PACKAGE_DIR/models/"

# Copy dataset files
cp data/audio_processing/*.py "$PACKAGE_DIR/data/audio_processing/"

# Copy training script
cp training_scripts/train_konkanivani_asr.py "$PACKAGE_DIR/training_scripts/"

# Copy checkpoint
if [ -f "archives/checkpoint_epoch_15.pt" ]; then
    cp archives/checkpoint_epoch_15.pt "$PACKAGE_DIR/checkpoints/"
    echo "✅ Checkpoint included"
fi

# Create zip
echo ""
echo "Creating zip file..."
cd "$PACKAGE_DIR"
zip -r ../kaggle_complete_dataset.zip . -q
cd ..

# Get size
SIZE=$(du -h kaggle_complete_dataset.zip | cut -f1)

echo ""
echo "=========================================="
echo "✅ Package created: kaggle_complete_dataset.zip"
echo "   Size: $SIZE"
echo "   Audio files: $AUDIO_COUNT"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Go to https://www.kaggle.com/datasets"
echo "  2. Click 'New Dataset'"
echo "  3. Upload: kaggle_complete_dataset.zip"
echo "  4. Title: 'Konkani ASR Complete Dataset'"
echo "  5. Make it Private"
echo "  6. Click 'Create'"
echo ""
echo "Then in your Kaggle notebook:"
echo "  1. Add this new dataset"
echo "  2. Update DATASET_PATH in Step 2"
echo "  3. Run training!"
echo ""
