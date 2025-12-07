#!/bin/bash

# KonkaniVani ASR Training - Memory Optimized for Tesla T4
# =========================================================

echo "🚀 Starting memory-optimized training for Tesla T4 GPU"
echo ""
echo "Memory optimizations enabled:"
echo "  ✅ Batch size: 2 (reduced from 8)"
echo "  ✅ Gradient accumulation: 4 steps (effective batch size = 8)"
echo "  ✅ Mixed precision (FP16)"
echo "  ✅ Smaller model: d_model=192, 8 encoder, 4 decoder layers"
echo "  ✅ Periodic CUDA cache clearing"
echo ""

# Clear any existing CUDA cache
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Run training with memory-optimized settings
python3 training_scripts/train_konkanivani_asr.py \
    --train_manifest data/konkani-asr-v0/splits/manifests/train.json \
    --val_manifest data/konkani-asr-v0/splits/manifests/val.json \
    --vocab_file data/vocab.json \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 50 \
    --learning_rate 0.0005 \
    --device cuda \
    --d_model 192 \
    --encoder_layers 8 \
    --decoder_layers 4 \
    --mixed_precision \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --resume checkpoints/checkpoint_epoch_15.pt

echo ""
echo "✅ Training completed!"
