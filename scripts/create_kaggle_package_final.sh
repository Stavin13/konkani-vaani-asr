#!/bin/bash

# Create optimized Kaggle package with correct audio path
# Uses: data/konkani-asr-v0/data/processed_segments_with_speakers/audio_segments

echo "📦 Creating Kaggle package with correct audio path..."
echo ""

# Clean up any previous attempts
rm -rf kaggle_package
rm -f kaggle_konkani_package.zip

# Create directory structure
mkdir -p kaggle_package

echo "✅ Step 1: Copying code files..."
cp -r training_scripts kaggle_package/
cp -r models kaggle_package/
mkdir -p kaggle_package/data/audio_processing
cp -r data/audio_processing/* kaggle_package/data/audio_processing/

echo "✅ Step 2: Copying manifests and vocab..."
mkdir -p kaggle_package/data/konkani-asr-v0/splits/manifests
cp data/konkani-asr-v0/splits/manifests/*.json kaggle_package/data/konkani-asr-v0/splits/manifests/
cp data/vocab.json kaggle_package/data/

echo "✅ Step 3: Copying checkpoint (294 MB)..."
mkdir -p kaggle_package/archives
cp archives/checkpoint_epoch_15.pt kaggle_package/archives/

echo "✅ Step 4: Copying audio files from correct location..."
AUDIO_SOURCE="data/konkani-asr-v0/data/processed_segments_with_speakers/audio_segments"

if [ -d "$AUDIO_SOURCE" ]; then
    echo "   Found audio at: $AUDIO_SOURCE"
    
    # Count files
    AUDIO_COUNT=$(find "$AUDIO_SOURCE" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | wc -l)
    echo "   Audio files found: $AUDIO_COUNT"
    
    # Create target directory
    mkdir -p kaggle_package/data/konkani-asr-v0/audio
    
    # Copy audio files
    echo "   Copying audio files (this may take a few minutes)..."
    cp -r "$AUDIO_SOURCE"/* kaggle_package/data/konkani-asr-v0/audio/
    
    echo "   ✅ Audio files copied successfully"
else
    echo "   ❌ Audio directory not found at: $AUDIO_SOURCE"
    echo "   Please check the path"
    exit 1
fi

echo ""
echo "✅ Step 5: Creating zip file..."
echo "   This may take 5-10 minutes for large audio files..."
zip -r -q kaggle_konkani_package.zip kaggle_package/

echo ""
echo "✅ Package created successfully!"
echo ""
echo "📊 Package details:"
ls -lh kaggle_konkani_package.zip

echo ""
echo "📁 Package contents:"
du -sh kaggle_package/*

echo ""
echo "📤 Next steps:"
echo "   1. Go to https://www.kaggle.com/datasets"
echo "   2. Click 'New Dataset'"
echo "   3. Upload: kaggle_konkani_package.zip"
echo "   4. Title: 'Konkani ASR Training Data'"
echo "   5. Make it private"
echo "   6. Create dataset"
echo ""
echo "⏱️  Upload time estimate:"
SIZE_MB=$(du -m kaggle_konkani_package.zip | cut -f1)
echo "   Size: ${SIZE_MB} MB"
if [ $SIZE_MB -lt 500 ]; then
    echo "   Upload time: ~5-10 minutes"
elif [ $SIZE_MB -lt 2000 ]; then
    echo "   Upload time: ~15-30 minutes"
else
    echo "   Upload time: ~30-60 minutes"
fi

echo ""
echo "💡 After upload, use KAGGLE_TRAINING.ipynb to start training!"
