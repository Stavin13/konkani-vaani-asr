#!/bin/bash
# Setup Script for Kaggle Translation & Emotion Training

echo "🚀 Setting up Kaggle Training for Translation & Emotion"
echo "=========================================================="

# Check if we have the necessary notebooks
echo ""
echo "📋 Checking notebooks..."
if [ -f "notebooks/KAGGLE_TRANSLATION_TRAINING.ipynb" ]; then
    echo "  ✅ Translation notebook found"
else
    echo "  ❌ Translation notebook missing"
fi

if [ -f "notebooks/KAGGLE_EMOTION_TRAINING.ipynb" ]; then
    echo "  ✅ Emotion notebook found"
else
    echo "  ❌ Emotion notebook missing"
fi

# Check for data
echo ""
echo "📊 Checking data..."
echo ""
echo "Do you have translation data? (Konkani-English pairs)"
read -p "  (y/n): " HAS_TRANSLATION

echo ""
echo "Do you have emotion data? (Konkani text with emotion labels)"
read -p "  (y/n): " HAS_EMOTION

# Generate data if needed
echo ""
if [[ $HAS_TRANSLATION =~ ^[Nn]$ ]]; then
    echo "📝 You need to generate translation data first:"
    echo "   Option 1: Run notebooks/GENERATE_TRANSLATION_DATA.ipynb"
    echo "   Option 2: Manually create data/translation/train.json"
    echo ""
fi

if [[ $HAS_EMOTION =~ ^[Nn]$ ]]; then
    echo "📝 You need to generate emotion data first:"
    echo "   Option 1: Run notebooks/GENERATE_EMOTION_DATA.ipynb"
    echo "   Option 2: Manually create data/emotion/train.csv"
    echo ""
fi

# Package data for Kaggle
echo "=========================================================="
echo ""
echo "📦 Ready to package data for Kaggle?"
read -p "   (y/n): " PACKAGE

if [[ $PACKAGE =~ ^[Yy]$ ]]; then
    echo ""
    echo "Packaging translation data..."
    if [ -d "data/translation" ]; then
        cd data/translation
        zip -r ../../konkani_translation_data.zip . -x "*.DS_Store" -x "._*"
        cd ../..
        echo "  ✅ Created konkani_translation_data.zip"
    else
        echo "  ⚠️  data/translation/ not found"
    fi
    
    echo ""
    echo "Packaging emotion data..."
    if [ -d "data/emotion" ]; then
        cd data/emotion
        zip -r ../../konkani_emotion_data.zip . -x "*.DS_Store" -x "._*"
        cd ../..
        echo "  ✅ Created konkani_emotion_data.zip"
    else
        echo "  ⚠️  data/emotion/ not found"
    fi
fi

# Summary
echo ""
echo "=========================================================="
echo "✨ Setup Complete!"
echo "=========================================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Upload datasets to Kaggle:"
echo "   • Go to https://www.kaggle.com/datasets"
echo "   • Click 'New Dataset'"
echo "   • Upload konkani_translation_data.zip"
echo "   • Upload konkani_emotion_data.zip"
echo ""
echo "2. Create Kaggle notebooks:"
echo "   • Go to https://www.kaggle.com/code"
echo "   • Click 'New Notebook'"
echo "   • Upload notebooks/KAGGLE_TRANSLATION_TRAINING.ipynb"
echo "   • Upload notebooks/KAGGLE_EMOTION_TRAINING.ipynb"
echo ""
echo "3. Add datasets to notebooks and run!"
echo ""
echo "📖 Full guide: docs/KAGGLE_PARALLEL_TRAINING_GUIDE.md"
echo ""
