#!/bin/bash

# Create minimal Kaggle package - FIXED VERSION
# Only includes essential files, not the full 14GB

echo "📦 Creating minimal Kaggle package (FIXED)..."
echo ""

# Create temp directory
rm -rf kaggle_minimal
mkdir -p kaggle_minimal

echo "✅ Step 1: Copying code files..."
cp -r training_scripts kaggle_minimal/
cp -r models kaggle_minimal/
cp -r data/audio_processing kaggle_minimal/data/audio_processing/

echo "✅ Step 2: Copying manifests and vocab..."
mkdir -p kaggle_minimal/data/konkani-asr-v0/splits/manifests
cp data/konkani-asr-v0/splits/manifests/*.json kaggle_minimal/data/konkani-asr-v0/splits/manifests/
cp data/vocab.json kaggle_minimal/data/

echo "✅ Step 3: Copying checkpoint..."
mkdir -p kaggle_minimal/archives
cp archives/checkpoint_epoch_15.pt kaggle_minimal/archives/

echo "✅ Step 4: Copying audio files..."
# Check if audio directory exists
if [ -d "data/konkani-asr-v0/audio" ]; then
    mkdir -p kaggle_minimal/data/konkani-asr-v0/audio
    cp -r data/konkani-asr-v0/audio/* kaggle_minimal/data/konkani-asr-v0/audio/
    echo "   ✅ Audio files copied"
else
    echo "   ⚠️  Audio directory not found at data/konkani-asr-v0/audio"
    echo "   Checking alternative locations..."
    
    # Try alternative paths
    if [ -d "data/audio" ]; then
        mkdir -p kaggle_minimal/data/konkani-asr-v0/audio
        cp -r data/audio/* kaggle_minimal/data/konkani-asr-v0/audio/
        echo "   ✅ Found and copied from data/audio"
    elif [ -d "KonkaniRawSpeechCorpus/Data" ]; then
        mkdir -p kaggle_minimal/data/konkani-asr-v0/audio
        cp -r KonkaniRawSpeechCorpus/Data/* kaggle_minimal/data/konkani-asr-v0/audio/
        echo "   ✅ Found and copied from KonkaniRawSpeechCorpus/Data"
    else
        echo "   ❌ Could not find audio files!"
        echo "   Please check where your audio files are located"
    fi
fi

echo ""
echo "✅ Step 5: Creating zip file..."
zip -r -q kaggle_konkani_minimal.zip kaggle_minimal/

echo ""
echo "✅ Done!"
echo ""
ls -lh kaggle_konkani_minimal.zip

echo ""
echo "📊 Package contents:"
du -sh kaggle_minimal/*

echo ""
echo "📤 Next steps:"
echo "   1. Upload kaggle_konkani_minimal.zip to Kaggle"
echo "   2. Create new dataset"
echo "   3. Use KAGGLE_TRAINING.ipynb"
echo ""
echo "💡 This should upload MUCH faster than 14GB!"
