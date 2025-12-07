#!/bin/bash

# Complete NER Training Pipeline
# Step 1: Auto-label data with pre-trained model
# Step 2: Train custom NER model

echo "=================================="
echo "KONKANI NER TRAINING PIPELINE"
echo "=================================="
echo ""

# Check if running on Mac or Linux/Colab
if [[ "$OSTYPE" == "darwin"* ]]; then
    DEVICE="mps"
    echo "🍎 Detected macOS - using MPS device"
else
    DEVICE="cuda"
    echo "🐧 Detected Linux - using CUDA device"
fi

echo ""
echo "Step 1: Auto-labeling data with pre-trained model"
echo "This will take ~10-15 minutes for 2,500 samples"
echo "--------------------------------------------------"

python3 scripts/auto_label_ner.py \
    --input transcripts_konkani_cleaned.json \
    --output data/ner_labeled_data.json

if [ $? -ne 0 ]; then
    echo "❌ Auto-labeling failed!"
    exit 1
fi

echo ""
echo "✅ Auto-labeling complete!"
echo ""
echo "Step 2: Training custom NER model"
echo "This will take ~2-3 hours on GPU"
echo "--------------------------------------------------"

python3 train_konkani_ner.py \
    --data_file data/ner_labeled_data.json \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --device $DEVICE \
    --checkpoint_dir checkpoints/ner

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ NER TRAINING COMPLETE!"
echo "=================================="
echo ""
echo "Model saved to: checkpoints/ner/best_ner_model.pt"
echo "Ready to integrate into complete system!"
