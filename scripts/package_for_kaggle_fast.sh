#!/bin/bash
# Fast Kaggle packaging - only essential files

echo "=========================================="
echo "Fast Kaggle Packaging"
echo "=========================================="

cd /Volumes/data\&proj/konkani

# Create output directory
mkdir -p kaggle_package_fast

echo ""
echo "Packaging ONLY essential files (no raw audio)..."
echo ""

# Package code, models, and existing processed data
zip -r kaggle_package_fast/konkani_code_only.zip \
    training_scripts/ \
    models/ \
    data/audio_processing/ \
    data/vocab.json \
    data/konkani-asr-v0/splits/manifests/ \
    config/ \
    scripts/prepare_raw_corpus_data.py \
    scripts/test_best_model.py \
    scripts/train_with_periodic_testing.py \
    -x "*.pyc" "*__pycache__*" "*.DS_Store" "*__MACOSX*"

echo ""
echo "✓ Created code package (small, fast)"
ls -lh kaggle_package_fast/konkani_code_only.zip

echo ""
echo "=========================================="
echo "IMPORTANT: Audio Data Strategy"
echo "=========================================="
echo ""
echo "Option 1: Use existing processed data (FASTEST)"
echo "  → Upload: konkani_code_only.zip"
echo "  → Size: ~10-50 MB"
echo "  → Data: 21 hours (existing manifests)"
echo "  → Time: 2 minutes"
echo ""
echo "Option 2: Add raw corpus separately (BETTER RESULTS)"
echo "  → First upload: konkani_code_only.zip"
echo "  → Then upload: KonkaniRawSpeechCorpus as separate dataset"
echo "  → Kaggle will handle the large files"
echo "  → Data: 88 hours total"
echo ""
echo "Option 3: Use your existing zip (IF IT HAS EVERYTHING)"
echo "  → Upload: /Users/stavinfernandes/kaggle_complete_dataset.zip"
echo "  → Check if it has KonkaniRawSpeechCorpus inside"
echo ""
echo "=========================================="
echo "Recommended: Option 1 (fastest to test fix)"
echo "=========================================="
echo ""
echo "The CTC weight fix (0.3 → 0.8) will help even with 21h data."
echo "You can always add more data later if needed."
echo ""
