#!/bin/bash
# Create lightweight package for Colab (code + manifests, no audio)

echo "📦 Creating Colab package..."

# Create the zip with essential code files + manifests
zip -r konkani_code.zip \
    models/ \
    data/audio_processing/ \
    data/konkani-asr-v0/splits/manifests/*.json \
    train_konkanivani_asr.py \
    inference_konkanivani.py \
    evaluate_konkanivani.py \
    vocab.json \
    -x "*.pyc" -x "__pycache__/*" -x ".DS_Store" -x "._*"

echo ""
echo "✅ Created: konkani_code.zip"
du -sh konkani_code.zip

echo ""
echo "📋 Contents:"
unzip -l konkani_code.zip | grep -E "(train\.json|val\.json|test\.json|\.py$)" | tail -10

echo ""
echo "📤 Next steps:"
echo "1. Upload this zip file in Colab (cell 3)"
echo "2. Audio files will be copied from your Drive in cell 4"
