#!/bin/bash
# Create a lightweight package for Colab (without audio files)

echo "Creating lightweight package for Colab..."

# Create temp directory
mkdir -p /tmp/konkani_colab

# Copy essential files
cp -r models /tmp/konkani_colab/
cp -r data/audio_processing /tmp/konkani_colab/data/
cp -r data/konkani-asr-v0/splits /tmp/konkani_colab/data/konkani-asr-v0/
cp train_konkanivani_asr.py /tmp/konkani_colab/
cp inference_konkanivani.py /tmp/konkani_colab/
cp evaluate_konkanivani.py /tmp/konkani_colab/
cp vocab.json /tmp/konkani_colab/
cp requirements.txt /tmp/konkani_colab/
cp KONKANIVANI_README.md /tmp/konkani_colab/

# Create zip (should be < 100MB)
cd /tmp
zip -r konkani_colab_lite.zip konkani_colab/

echo "✅ Created: /tmp/konkani_colab_lite.zip"
du -sh /tmp/konkani_colab_lite.zip

echo ""
echo "Next steps:"
echo "1. Upload this zip to Google Drive"
echo "2. Audio files will be downloaded separately in Colab"
