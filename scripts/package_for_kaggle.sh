#!/bin/bash
# Package project for Kaggle upload (TAR Version)

echo "=========================================="
echo "Packaging for Kaggle (TAR.GZ)"
echo "=========================================="

# Create output directory
mkdir -p kaggle_package
cd kaggle_package

###############################################
# 1. TAR: Konkani Raw Speech Corpus
###############################################
echo ""
echo "1. Packaging KonkaniRawSpeechCorpus (tar.gz)..."

if [ -d "../KonkaniRawSpeechCorpus" ]; then
    
    tar --exclude="*.DS_Store" \
        --exclude="__MACOSX" \
        -cvzf konkani_raw_corpus.tar.gz \
        -C .. KonkaniRawSpeechCorpus

    echo "   ✓ Created konkani_raw_corpus.tar.gz"
    ls -lh konkani_raw_corpus.tar.gz
else
    echo "   ✗ KonkaniRawSpeechCorpus not found"
fi


###############################################
# 2. TAR: Code + Processed Data
###############################################
echo ""
echo "2. Packaging existing data and code (tar.gz)..."

tar --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="*.DS_Store" \
    --exclude="__MACOSX" \
    -cvzf konkani_code_data.tar.gz \
    -C .. data/konkani-asr-v0 \
          data/vocab.json \
          models \
          data/audio_processing

echo "   ✓ Created konkani_code_data.tar.gz"
ls -lh konkani_code_data.tar.gz


###############################################
# 3. TAR: Combined Dataset (Full Package)
###############################################
echo ""
echo "3. Creating combined package (tar.gz)..."

tar --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="*.DS_Store" \
    --exclude="__MACOSX" \
    --exclude="*.git" \
    -cvzf konkani_complete_data.tar.gz \
    -C .. KonkaniRawSpeechCorpus \
          data/konkani-asr-v0 \
          data/vocab.json \
          models \
          data/audio_processing

echo "   ✓ Created konkani_complete_data.tar.gz"
ls -lh konkani_complete_data.tar.gz


###############################################
# Summary
###############################################
echo ""
echo "=========================================="
echo "Package Summary"
echo "=========================================="
ls -lh *.tar.gz
echo ""

total_size=$(du -sh . | cut -f1)
echo "Total size: $total_size"


###############################################
# Upload Instructions
###############################################
echo ""
echo "=========================================="
echo "Upload Instructions"
echo "=========================================="
echo ""
echo "Option 1: Upload complete package (if < 20GB)"
echo "  → Upload: konkani_complete_data.tar.gz"
echo ""
echo "Option 2: Upload separately (if > 20GB)"
echo "  → Upload: konkani_raw_corpus.tar.gz (Dataset 1)"
echo "  → Upload: konkani_code_data.tar.gz (Dataset 2)"
echo ""
echo "Next steps:"
echo "  1. Go to https://www.kaggle.com/datasets"
echo "  2. Click 'New Dataset'"
echo "  3. Upload the tar.gz file(s)"
echo "  4. Make it Private"
echo "  5. Follow: docs/KAGGLE_RETRAIN_GUIDE.md"
echo ""
echo "✓ Packaging complete (TAR version)!"
