#!/usr/bin/env python3
"""
Resume training from a PyTorch checkpoint and fine-tune the model further
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import yaml
import argparse
from pathlib import Path
import os
import sys
from tqdm import tqdm
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '.')

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"resume_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_checkpoint_info(checkpoint_path):
    """Load and inspect checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint contents:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: Dict with {len(value)} keys")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    # Extract training info
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    train_loss = checkpoint.get('train_loss', float('inf'))
    
    print(f"\nTraining state:")
    print(f"  Last epoch: {epoch}")
    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Training loss: {train_loss:.4f}")
    
    return checkpoint

def create_model_from_checkpoint(checkpoint, device='cuda'):
    """Create model from checkpoint"""
    try:
        # Try to import the model
        from models.konkanivani_asr import create_konkanivani_model
        
        # Get model config from checkpoint
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        print(f"Model config: {model_config}")
        
        # Create model with config from checkpoint
        model = create_konkanivani_model(
            vocab_size=model_config.get('vocab_size', 81),
            d_model=model_config.get('d_model', 128),
            encoder_layers=model_config.get('encoder_layers', 8),
            decoder_layers=model_config.get('decoder_layers', 6),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.3),
            conv_kernel_size=model_config.get('conv_kernel_size', 31)
        )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Warning: No model state dict found in checkpoint")
        
        model = model.to(device)
        print(f"✓ Model loaded successfully on {device}")
        
        return model, model_config
        
    except ImportError as e:
        print(f"Error importing model: {e}")
        print("Make sure the models directory is in your Python path")
        return None, None

def create_optimizer_and_scheduler(model, checkpoint, config):
    """Create optimizer and scheduler, loading state if available"""
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )
    
    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
    
    return optimizer, scheduler

def create_training_config(checkpoint_path, additional_epochs=50):
    """Create training configuration for resuming"""
    
    config = {
        'model': {
            'vocab_size': 81,
            'd_model': 128,
            'encoder_layers': 8,
            'decoder_layers': 6,
            'num_heads': 4,
            'conv_kernel_size': 31,
            'dropout': 0.3
        },
        'training': {
            'learning_rate': 0.00005,  # Reduced for fine-tuning
            'weight_decay': 0.0001,
            'grad_clip': 5.0,
            'ctc_weight': 0.9,
            'batch_size': 8,
            'gradient_accumulation_steps': 2,
            'mixed_precision': True,
            'additional_epochs': additional_epochs,
            'save_every': 5,
            'test_every': 5
        },
        'data': {
            'train_manifest': 'konkani-10k/train_manifest.json',
            'val_manifest': 'konkani-10k/val_manifest.json',
            'vocab_file': 'konkani-10k/vocab.json',
            'num_workers': 2
        },
        'paths': {
            'checkpoint_dir': 'checkpoints_resumed',
            'log_dir': 'logs_resumed',
            'resume_from': str(checkpoint_path)
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    return config

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, checkpoint_dir):
    """Save training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save as best model if it's the best validation loss
    best_model_path = checkpoint_dir / 'best_model_resumed.pt'
    if not best_model_path.exists() or val_loss < get_best_val_loss(checkpoint_dir):
        torch.save(checkpoint, best_model_path)
        print(f"✓ Saved new best model with val_loss: {val_loss:.4f}")
    
    return checkpoint_path

def get_best_val_loss(checkpoint_dir):
    """Get the best validation loss from existing checkpoints"""
    best_model_path = Path(checkpoint_dir) / 'best_model_resumed.pt'
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            return checkpoint.get('val_loss', float('inf'))
        except:
            pass
    return float('inf')

def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--additional_epochs', type=int, default=50,
                       help='Number of additional epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='resumed_training',
                       help='Output directory for resumed training')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting resumed training from {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = load_checkpoint_info(args.checkpoint)
    
    # Create model
    model, model_config = create_model_from_checkpoint(checkpoint, device)
    if model is None:
        logger.error("Failed to create model")
        return
    
    # Create training config
    config = create_training_config(args.checkpoint, args.additional_epochs)
    config['training']['learning_rate'] = args.learning_rate
    config['training']['batch_size'] = args.batch_size
    config['paths']['checkpoint_dir'] = str(checkpoint_dir)
    config['paths']['log_dir'] = str(log_dir)
    config['device'] = device
    
    # Save config
    config_path = output_dir / 'resume_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {config_path}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, checkpoint, config)
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    logger.info(f"Resuming training from epoch {start_epoch}")
    
    print(f"""
=================================================================
RESUME TRAINING SETUP COMPLETE
=================================================================
Checkpoint: {args.checkpoint}
Starting epoch: {start_epoch}
Additional epochs: {args.additional_epochs}
Learning rate: {config['training']['learning_rate']}
Device: {device}
Output directory: {output_dir}
=================================================================

To continue with actual training, you'll need to:
1. Implement the data loading logic
2. Implement the training loop
3. Run the training

This script has prepared everything for resuming training!
""")
    
    # Create a simple training script template
    training_script = f"""
# Training loop template - customize as needed
import torch
from torch.utils.data import DataLoader

# Load your datasets here
# train_dataset = YourDataset(config['data']['train_manifest'])
# val_dataset = YourDataset(config['data']['val_manifest'])
# train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'])
# val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

# Training loop
for epoch in range({start_epoch}, {start_epoch + args.additional_epochs}):
    # Training phase
    model.train()
    train_loss = 0.0
    
    # for batch in train_loader:
    #     # Your training logic here
    #     pass
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    # for batch in val_loader:
    #     # Your validation logic here
    #     pass
    
    # Save checkpoint
    # save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, config, checkpoint_dir)
    
    print(f"Epoch {{epoch}}: Train Loss: {{train_loss:.4f}}, Val Loss: {{val_loss:.4f}}")
"""
    
    training_template_path = output_dir / 'training_template.py'
    with open(training_template_path, 'w') as f:
        f.write(training_script)
    
    logger.info(f"Training template saved to {training_template_path}")

if __name__ == "__main__":
    main()