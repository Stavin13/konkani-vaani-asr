#!/bin/bash

# Resume KonkaniVani ASR Training from Checkpoint 15
# ===================================================
# This script uses the EXACT configuration from checkpoint_epoch_15.pt
# with memory optimizations to prevent OOM errors on Tesla T4

echo "======================================================================"
echo "🚀 RESUMING KONKANIVANI ASR TRAINING FROM CHECKPOINT 15"
echo "======================================================================"
echo ""
echo "📋 Configuration (from checkpoint_epoch_15.pt):"
echo "   • Model: d_model=256, 12 encoder layers, 6 decoder layers"
echo "   • Vocab size: 200 characters"
echo "   • Learning rate: 0.0005"
echo "   • CTC weight: 0.3"
echo ""
echo "💾 Memory Optimizations for Tesla T4 (14GB):"
echo "   • Batch size: 2 (reduced from 8)"
echo "   • Gradient accumulation: 4 steps (effective batch = 8)"
echo "   • Mixed precision: FP16 enabled"
echo "   • Periodic CUDA cache clearing"
echo ""
echo "📂 Resuming from: archives/checkpoint_epoch_15.pt"
echo "======================================================================"
echo ""

# Copy checkpoint to checkpoints directory if not already there
if [ ! -f "checkpoints/checkpoint_epoch_15.pt" ]; then
    echo "📋 Copying checkpoint to checkpoints directory..."
    cp archives/checkpoint_epoch_15.pt checkpoints/
    echo "✅ Checkpoint copied"
    echo ""
fi

# Clear any existing CUDA cache
echo "🧹 Clearing CUDA cache..."
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('CUDA not available')"
echo ""

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "🎯 Starting training..."
echo ""

# Run training with EXACT config from checkpoint + memory optimizations
python3 training_scripts/train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file data/vocab.json \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 50 \
    --learning_rate 0.0005 \
    --device cuda \
    --d_model 256 \
    --encoder_layers 12 \
    --decoder_layers 6 \
    --mixed_precision \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --resume checkpoints/checkpoint_epoch_15.pt

echo ""
echo "======================================================================"
echo "✅ TRAINING COMPLETED!"
echo "======================================================================"
