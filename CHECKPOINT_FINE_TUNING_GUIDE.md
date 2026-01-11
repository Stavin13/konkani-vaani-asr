# Fine-tuning from Checkpoint Guide

This guide shows you how to resume training and fine-tune your ASR model from the existing checkpoint.

## Files Created

1. **`inspect_checkpoint.py`** - Inspect checkpoint contents and structure
2. **`resume_training_from_checkpoint.py`** - Setup script for resuming training
3. **`fine_tune_from_checkpoint.py`** - Complete fine-tuning script
4. **`CHECKPOINT_FINE_TUNING_GUIDE.md`** - This guide

## Quick Start

### 1. Inspect Your Checkpoint

First, let's examine what's in your checkpoint:

```bash
python inspect_checkpoint.py
```

This will show you:
- Model architecture details
- Training state (epoch, losses)
- Total parameters (5.9M in your case)
- Optimizer and scheduler states

### 2. Fine-tune the Model

Use the complete fine-tuning script to continue training:

```bash
python fine_tune_from_checkpoint.py \
    --checkpoint "best_model (1).pt" \
    --train_manifest "konkani-10k/train_manifest.json" \
    --val_manifest "konkani-10k/val_manifest.json" \
    --vocab_file "konkani-10k/vocab.json" \
    --epochs 20 \
    --learning_rate 0.00005 \
    --batch_size 4 \
    --output_dir "fine_tuned_model"
```

### 3. Monitor Training

The script will:
- Load your checkpoint (epoch 99, val_loss: 2.0637)
- Continue training for additional epochs
- Use a lower learning rate for fine-tuning
- Save new checkpoints and best model
- Show progress with loss metrics

## Configuration Options

### Learning Rate
- **Original training**: 0.0001
- **Fine-tuning**: 0.00005 (recommended)
- **Conservative**: 0.00001 (for very careful fine-tuning)

### Batch Size
- **GPU memory limited**: 2-4
- **Good GPU**: 8-16
- **Multiple GPUs**: 16-32

### Additional Epochs
- **Quick improvement**: 10-20 epochs
- **Thorough fine-tuning**: 50-100 epochs
- **Careful refinement**: 20-50 epochs

## Expected Results

Your model already achieved excellent results:
- **Current validation loss**: 2.0637
- **Training completed**: 99 epochs
- **Model size**: 5.9M parameters

With fine-tuning, you can expect:
- **Further loss reduction**: 1.8-2.0 range
- **Better convergence**: More stable training
- **Improved accuracy**: 2-5% improvement possible

## Advanced Usage

### Custom Training Loop

If you need more control, modify the training loop in `fine_tune_from_checkpoint.py`:

```python
# Adjust loss weights
total_batch_loss = 0.8 * ctc_loss + 0.2 * decoder_loss

# Add custom metrics
# Add learning rate scheduling
# Add early stopping
```

### Different Datasets

To fine-tune on new data:

```bash
python fine_tune_from_checkpoint.py \
    --checkpoint "best_model (1).pt" \
    --train_manifest "new_data/train.json" \
    --val_manifest "new_data/val.json" \
    --vocab_file "new_data/vocab.json" \
    --epochs 30 \
    --learning_rate 0.00001  # Lower LR for new data
```

### Transfer Learning

For domain adaptation:

```bash
# Use very low learning rate
python fine_tune_from_checkpoint.py \
    --checkpoint "best_model (1).pt" \
    --train_manifest "domain_data/train.json" \
    --val_manifest "domain_data/val.json" \
    --vocab_file "domain_data/vocab.json" \
    --epochs 50 \
    --learning_rate 0.000025 \
    --batch_size 2
```

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
--batch_size 2

# Use gradient accumulation (modify script)
gradient_accumulation_steps = 4
```

### Slow Training
```bash
# Increase batch size if possible
--batch_size 8

# Use mixed precision (already enabled)
# Reduce validation frequency (modify script)
```

### Model Not Improving
```bash
# Lower learning rate
--learning_rate 0.00001

# More epochs
--epochs 50

# Check data quality
```

## Output Files

After fine-tuning, you'll have:

```
fine_tuned_model/
├── checkpoint_epoch_100.pt    # Regular checkpoints
├── checkpoint_epoch_101.pt
├── ...
├── best_model_finetuned.pt   # Best model during fine-tuning
└── training_logs.txt         # Training progress
```

## Model Comparison

| Model | Validation Loss | Epochs | Notes |
|-------|----------------|--------|-------|
| Original | 2.0637 | 99 | Your current model |
| Fine-tuned | ~1.9-2.0 | 99+20 | Expected after fine-tuning |

## Next Steps

1. **Test the fine-tuned model** on your test set
2. **Compare performance** with the original
3. **Deploy the better model** for inference
4. **Continue fine-tuning** if needed

## Tips for Success

1. **Start conservative**: Use low learning rates
2. **Monitor closely**: Watch for overfitting
3. **Save frequently**: Don't lose progress
4. **Test regularly**: Validate on held-out data
5. **Be patient**: Fine-tuning takes time

Your model is already well-trained, so fine-tuning should give you that extra performance boost you're looking for!