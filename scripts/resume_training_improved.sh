#!/bin/bash
# Resume training with improved settings for better CTC predictions

echo "Resuming ASR training with improved CTC settings..."

python training_scripts/train_konkanivani_asr.py \
  --resume_from kaggle_asr_outputs/checkpoints/checkpoint_epoch_50.pt \
  --epochs 100 \
  --ctc_weight 0.7 \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --checkpoint_dir checkpoints \
  --log_dir logs \
  --device mps

echo "Training complete!"
