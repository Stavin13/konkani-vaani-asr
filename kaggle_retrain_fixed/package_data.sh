#!/bin/bash
# Package data for Kaggle upload

echo "📦 Creating Kaggle data package..."

# Create data directory
mkdir -p kaggle_data_package

# Copy training manifests
cp data/konkani-asr-v0/splits/manifests/train.json kaggle_data_package/
cp data/konkani-asr-v0/splits/manifests/val.json kaggle_data_package/
cp data/konkani-asr-v0/splits/manifests/test.json kaggle_data_package/

# Copy audio data (this might be large!)
# Adjust paths based on your data structure
echo "⚠️  Audio data copying - this may take a while..."
cp -r data/konkani-asr-v0/data/processed_segments_diarized/audio_segments kaggle_data_package/

# Create zip file
echo "🗜️  Creating zip file..."
zip -r konkani_training_data.zip kaggle_data_package/

echo "✅ Data package ready: konkani_training_data.zip"
echo "📤 Upload this to Kaggle as a dataset"
