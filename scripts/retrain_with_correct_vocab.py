#!/usr/bin/env python3
"""
Retrain ASR Model with Correct Vocabulary Size (193 instead of 81)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR

def create_correct_vocabulary():
    """Create vocabulary with correct size from training data"""
    
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    # Count characters
    char_counter = Counter()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get('text', '')
            for char in text:
                char_counter[char] += 1
    
    # Create vocabulary
    vocab = {
        '<blank>': 0,  # CTC blank (MUST be 0)
        '<unk>': 1,    # Unknown token
    }
    
    # Add characters by frequency
    for char, count in char_counter.most_common():
        if char not in vocab:
            vocab[char] = len(vocab)
    
    print(f"Created vocabulary with {len(vocab)} tokens")
    return vocab

def create_model_with_correct_vocab_size():
    """Create model with correct vocabulary size"""
    
    vocab = create_correct_vocabulary()
    vocab_size = len(vocab)
    
    print(f"Creating model with vocab_size={vocab_size}")
    
    # Create model with correct vocabulary size
    model = KonkaniVaniASR(
        vocab_size=vocab_size,  # 193 instead of 81
        d_model=256,
        encoder_layers=12,
        dropout=0.1
    )
    
    print(f"Model created:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save initial checkpoint with correct vocabulary
    checkpoint_dir = Path('checkpoints/corrected_vocab')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    initial_checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': {
            'vocab_size': vocab_size,
            'd_model': 256,
            'encoder_layers': 12,
            'dropout': 0.1
        }
    }
    
    checkpoint_path = checkpoint_dir / 'initial_model_correct_vocab.pt'
    torch.save(initial_checkpoint, checkpoint_path)
    
    print(f"✅ Initial model saved: {checkpoint_path}")
    print(f"✅ Ready for training with correct vocabulary!")
    
    return model, vocab

def create_training_config():
    """Create training configuration for corrected model"""
    
    config = {
        'model': {
            'vocab_size': 193,  # Correct size!
            'd_model': 256,
            'encoder_layers': 12,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'num_epochs': 100,
            'save_every': 5,
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
        },
        'data': {
            'train_manifest': 'data/konkani-asr-v0/splits/manifests/train.json',
            'val_manifest': 'data/konkani-asr-v0/splits/manifests/val.json',
            'sample_rate': 16000,
            'n_mels': 80
        }
    }
    
    config_path = Path('config/corrected_vocab_training.json')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Training config saved: {config_path}")
    return config

def main():
    print("="*70)
    print("CREATING MODEL WITH CORRECT VOCABULARY SIZE")
    print("="*70)
    
    # Create model with correct vocab size
    model, vocab = create_model_with_correct_vocab_size()
    
    # Create training config
    config = create_training_config()
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Your model now has the correct vocabulary size (193)")
    print("2. Train the model:")
    print("   python training_scripts/train_konkanivani_asr.py \\")
    print("     --config config/corrected_vocab_training.json \\")
    print("     --resume checkpoints/corrected_vocab/initial_model_correct_vocab.pt")
    print()
    print("3. Or use Kaggle for faster training:")
    print("   - Upload the corrected model to Kaggle")
    print("   - Use the training notebook with vocab_size=193")
    print()
    print("Expected results with correct vocabulary:")
    print("  - Should see actual Devanagari characters in predictions")
    print("  - Accuracy should improve significantly (20-50%+)")
    print("  - Model will be able to predict all characters in your data")

if __name__ == '__main__':
    main()