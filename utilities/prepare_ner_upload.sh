#!/bin/bash

# Prepare files for NER training on Colab
# Creates a zip file with all necessary files

echo "=================================="
echo "PREPARING NER FILES FOR COLAB"
echo "=================================="
echo ""

# Check if files exist
echo "🔍 Checking files..."

FILES=(
    "transcripts_konkani_cleaned.json"
    "scripts/auto_label_ner.py"
    "models/konkani_ner.py"
    "train_konkani_ner.py"
)

ALL_EXIST=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - NOT FOUND"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "❌ Some files are missing!"
    exit 1
fi

echo ""
echo "📦 Creating zip file..."

# Create zip
zip -r ner_files.zip \
    transcripts_konkani_cleaned.json \
    scripts/auto_label_ner.py \
    models/konkani_ner.py \
    train_konkani_ner.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Created: ner_files.zip"
    echo ""
    echo "📊 File size:"
    ls -lh ner_files.zip
    echo ""
    echo "=================================="
    echo "READY FOR UPLOAD!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "  1. Go to https://colab.research.google.com"
    echo "  2. Upload train_ner_colab.ipynb"
    echo "  3. In Cell 3, upload ner_files.zip"
    echo "  4. Run all cells!"
    echo ""
else
    echo ""
    echo "❌ Failed to create zip file"
    exit 1
fi
