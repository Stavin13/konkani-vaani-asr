#!/bin/bash
# Package complete dataset for Kaggle upload
# This creates a zip with all necessary files for training

set -e

OUTPUT_ZIP="kaggle_complete_dataset.zip"
TEMP_DIR="kaggle_package_temp"

echo "=================================================="
echo "Packaging Complete Kaggle Dataset"
echo "=================================================="

# Clean up old package
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo ""
echo "1. Copying training scripts..."
mkdir -p "$TEMP_DIR/training_scripts"
cp training_scripts/train_konkanivani_asr.py "$TEMP_DIR/training_scripts/"

echo "2. Copying model files..."
mkdir -p "$TEMP_DIR/models"
cp models/konkanivani_asr.py "$TEMP_DIR/models/"

echo "3. Copying data processing..."
mkdir -p "$TEMP_DIR/data/audio_processing"
cp data/audio_processing/audio_processor.py "$TEMP_DIR/data/audio_processing/"
cp data/audio_processing/dataset.py "$TEMP_DIR/data/audio_processing/"

echo "4. Copying utility scripts..."
mkdir -p "$TEMP_DIR/scripts"
cp scripts/test_best_model.py "$TEMP_DIR/scripts/" 2>/dev/null || echo "  (test_best_model.py not found, skipping)"
cp scripts/generate_training_visualization.py "$TEMP_DIR/scripts/"

echo "5. Copying vocabulary..."
if [ -f "data/vocab.json" ]; then
    cp data/vocab.json "$TEMP_DIR/data/"
    echo "  ✓ vocab.json copied"
else
    echo "  ✗ WARNING: data/vocab.json not found!"
fi

echo "6. Copying manifest files..."
if [ -d "data/konkani-asr-v0/splits/manifests" ]; then
    mkdir -p "$TEMP_DIR/data/konkani-asr-v0/splits/manifests"
    cp data/konkani-asr-v0/splits/manifests/*.json "$TEMP_DIR/data/konkani-asr-v0/splits/manifests/"
    echo "  ✓ Manifests copied from konkani-asr-v0"
elif [ -d "data/konkani-combined/manifests" ]; then
    mkdir -p "$TEMP_DIR/data/konkani-combined/manifests"
    cp data/konkani-combined/manifests/*.json "$TEMP_DIR/data/konkani-combined/manifests/"
    echo "  ✓ Manifests copied from konkani-combined"
else
    echo "  ✗ WARNING: No manifest files found!"
    echo "    Expected: data/konkani-asr-v0/splits/manifests/ OR data/konkani-combined/manifests/"
fi

echo "7. Copying audio files..."
if [ -d "data/audio" ]; then
    echo "  Copying audio directory (this may take a while)..."
    cp -r data/audio "$TEMP_DIR/data/"
    AUDIO_COUNT=$(find "$TEMP_DIR/data/audio" -name "*.wav" | wc -l)
    echo "  ✓ Copied $AUDIO_COUNT audio files"
elif [ -d "data/konkani-asr-v0/audio" ]; then
    echo "  Copying audio from konkani-asr-v0..."
    mkdir -p "$TEMP_DIR/data/konkani-asr-v0"
    cp -r data/konkani-asr-v0/audio "$TEMP_DIR/data/konkani-asr-v0/"
    AUDIO_COUNT=$(find "$TEMP_DIR/data/konkani-asr-v0/audio" -name "*.wav" | wc -l)
    echo "  ✓ Copied $AUDIO_COUNT audio files"
else
    echo "  ✗ WARNING: No audio directory found!"
fi

echo ""
echo "8. Creating zip file..."
cd "$TEMP_DIR"
zip -r "../$OUTPUT_ZIP" . -q
cd ..

# Cleanup
rm -rf "$TEMP_DIR"

# Show results
FILE_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
echo ""
echo "=================================================="
echo "✓ Package created: $OUTPUT_ZIP ($FILE_SIZE)"
echo "=================================================="
echo ""
echo "Contents:"
unzip -l "$OUTPUT_ZIP" | head -30
echo ""
echo "Next steps:"
echo "1. Upload $OUTPUT_ZIP to Kaggle as a dataset"
echo "2. Create a new notebook on Kaggle"
echo "3. Add your dataset as input"
echo "4. Run the notebook!"
echo ""
