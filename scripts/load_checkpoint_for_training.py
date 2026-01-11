#!/usr/bin/env python3
"""
Load a checkpoint and set it to training mode for continued training
"""
import torch
import json
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.konkanivani_asr import KonkaniVaniASR


def load_checkpoint_for_training(checkpoint_path, device=None):
    """
    Load a checkpoint and return model in training mode
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (auto-detect if None)
    
    Returns:
        model: Model in training mode
        optimizer: Optimizer state (if available)
        scheduler: Scheduler state (if available) 
        epoch: Last epoch number
        vocab_info: Vocabulary mappings
    """
    
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    device = torch.device(device)
    print(f"🔧 Loading checkpoint on device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        checkpoint['model_state_dict'] = state_dict
    
    # Get vocab_size from checkpoint or infer from model
    if 'vocab_size' in checkpoint:
        vocab_size = checkpoint['vocab_size']
    else:
        # Infer from CTC head
        ctc_key = 'ctc_head.weight'
        vocab_size = state_dict[ctc_key].shape[0]
    
    # Infer model configuration
    encoder_key = 'encoder.input_proj.weight'
    d_model = state_dict[encoder_key].shape[0]
    
    # Count encoder layers
    encoder_layers = 0
    for key in state_dict.keys():
        if 'encoder.layers.' in key and '.ff1.0.weight' in key:
            layer_num = int(key.split('.')[2])
            encoder_layers = max(encoder_layers, layer_num + 1)
    
    # Count decoder layers  
    decoder_layers = 0
    for key in state_dict.keys():
        if 'decoder.decoder.layers.' in key and '.linear1.weight' in key:
            layer_num = int(key.split('.')[3])
            decoder_layers = max(decoder_layers, layer_num + 1)
    
    print(f"📊 Model config: vocab_size={vocab_size}, d_model={d_model}")
    print(f"📊 Architecture: encoder_layers={encoder_layers}, decoder_layers={decoder_layers}")
    
    # Create model
    model = KonkaniVaniASR(
        vocab_size=vocab_size,
        input_dim=80,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers
    )
    
    # Load model state
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Set to training mode
    model.train()
    print("🎯 Model set to training mode")
    
    # Extract training state
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    scheduler_state = checkpoint.get('scheduler_state_dict', None)
    epoch = checkpoint.get('epoch', 0)
    
    # Extract vocabulary
    vocab_info = {}
    if 'idx_to_char' in checkpoint and 'char_to_idx' in checkpoint:
        vocab_info['idx_to_char'] = checkpoint['idx_to_char']
        vocab_info['char_to_idx'] = checkpoint['char_to_idx']
    else:
        # Load from vocab.json
        vocab_paths = [
            project_root / "data" / "vocab.json",
            project_root / "data" / "konkani-10k" / "vocab.json",
        ]
        
        vocab_file = None
        for vp in vocab_paths:
            if vp.exists():
                vocab_file = vp
                break
        
        if vocab_file:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            vocab_info['char_to_idx'] = vocab_data['char2idx']
            vocab_info['idx_to_char'] = {int(k): v for k, v in vocab_data['idx2char'].items()}
    
    print(f"📚 Vocabulary size: {len(vocab_info.get('char_to_idx', {}))}")
    print(f"📈 Last epoch: {epoch}")
    
    return {
        'model': model,
        'optimizer_state': optimizer_state,
        'scheduler_state': scheduler_state,
        'epoch': epoch,
        'vocab_info': vocab_info,
        'device': device
    }


def setup_optimizer_and_scheduler(model, optimizer_state=None, scheduler_state=None):
    """
    Setup optimizer and scheduler for continued training
    
    Args:
        model: The model to optimize
        optimizer_state: Previous optimizer state (optional)
        scheduler_state: Previous scheduler state (optional)
    
    Returns:
        optimizer: Configured optimizer
        scheduler: Configured scheduler
    """
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Load previous optimizer state if available
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("✅ Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load optimizer state: {e}")
            print("🔄 Using fresh optimizer")
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Load previous scheduler state if available
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            print("✅ Loaded scheduler state from checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load scheduler state: {e}")
            print("🔄 Using fresh scheduler")
    
    return optimizer, scheduler


def main():
    """Example usage"""
    
    # Available checkpoints
    checkpoints = [
        "kaggle_asr_outputs/checkpoints/best_model.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_50.pt",
        "checkpoints/best_model_scripts1_fixed.pt",
        "checkpoints/checkpoint_epoch_15.pt",
        "best_model.pt",
    ]
    
    print("🔍 Available checkpoints:")
    for i, cp in enumerate(checkpoints):
        cp_path = Path(cp)
        exists = "✅" if cp_path.exists() else "❌"
        print(f"  {i+1}. {exists} {cp}")
    
    # Use the best available checkpoint
    checkpoint_path = None
    for cp in checkpoints:
        if Path(cp).exists():
            checkpoint_path = cp
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoints found!")
        return
    
    print(f"\n🚀 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint for training
    training_setup = load_checkpoint_for_training(checkpoint_path)
    
    model = training_setup['model']
    optimizer_state = training_setup['optimizer_state']
    scheduler_state = training_setup['scheduler_state']
    epoch = training_setup['epoch']
    vocab_info = training_setup['vocab_info']
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, optimizer_state, scheduler_state
    )
    
    print(f"\n✅ Model ready for training!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"📈 Starting from epoch: {epoch + 1}")
    
    # Example: Check model is in training mode
    print(f"🎯 Training mode: {model.training}")
    
    # You can now use this model for continued training
    # Example training loop would go here...
    
    return model, optimizer, scheduler, epoch, vocab_info


if __name__ == "__main__":
    main()