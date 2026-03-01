#!/usr/bin/env python3
"""
Create final model from checkpoint with optimized training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, '.')

def load_and_prepare_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint and prepare for final training"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Display checkpoint info
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', 'N/A')
    train_loss = checkpoint.get('train_loss', 'N/A')
    
    print(f"Checkpoint Info:")
    print(f"  Epoch: {epoch}")
    print(f"  Validation Loss: {val_loss}")
    print(f"  Training Loss: {train_loss}")
    
    try:
        from models.konkanivani_asr import create_konkanivani_model
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model
        model = create_konkanivani_model(
            vocab_size=model_config.get('vocab_size', 192),  # Updated for mega dataset
            d_model=model_config.get('d_model', 128),
            encoder_layers=model_config.get('encoder_layers', 8),
            decoder_layers=model_config.get('decoder_layers', 6),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.2),  # Reduced for final training
            conv_kernel_size=model_config.get('conv_kernel_size', 31)
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        
        model = model.to(device)
        print(f"✓ Model loaded successfully on {device}")
        
        return model, checkpoint, model_config
        
    except ImportError as e:
        print(f"Error importing model: {e}")
        return None, None, None

def create_final_training_config():
    """Create optimized config for final model training"""
    return {
        'learning_rate': 0.00003,  # Very low for final refinement
        'weight_decay': 0.0001,
        'batch_size': 16,  # Larger batch for stability
        'epochs': 20,  # Focused training
        'grad_clip': 3.0,
        'patience': 5,  # Early stopping
        'save_every': 5,
        'mixed_precision': True
    }

def save_final_model(model, checkpoint, config, output_path, metadata=None):
    """Save the final production-ready model"""
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': 'konkanivani_asr_final',
        'version': '1.0',
        'vocab_size': config.get('vocab_size', 192),
        'created_from_epoch': checkpoint.get('epoch', 0),
        'base_val_loss': checkpoint.get('val_loss', 'N/A'),
        'metadata': metadata or {}
    }
    
    # Save the model
    torch.save(final_checkpoint, output_path)
    print(f"✓ Final model saved to: {output_path}")
    
    # Create a deployment-ready version (model only)
    deployment_path = output_path.parent / f"{output_path.stem}_deployment.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': config.get('vocab_size', 192)
    }, deployment_path)
    print(f"✓ Deployment model saved to: {deployment_path}")
    
    return final_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Create final model from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='final_models',
                       help='Output directory for final model')
    parser.add_argument('--model_name', type=str, default='konkanivani_final',
                       help='Name for the final model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip additional training, just save current model as final')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, checkpoint, model_config = load_and_prepare_model(args.checkpoint, device)
    if model is None:
        print("Failed to load model")
        return
    
    # Create final model path
    final_model_path = output_dir / f"{args.model_name}.pt"
    
    if args.skip_training:
        # Just save current model as final
        print("Skipping additional training, saving current model as final...")
        
        metadata = {
            'source_checkpoint': str(args.checkpoint),
            'processing_date': str(Path().cwd()),
            'notes': 'Direct conversion from checkpoint without additional training'
        }
        
        save_final_model(model, checkpoint, model_config, final_model_path, metadata)
        
    else:
        # Perform final training (you would implement this)
        print("Additional training not implemented in this script.")
        print("Use fine_tune_from_checkpoint.py for additional training.")
        print("Or use --skip_training to save current model as final.")
        return
    
    # Create model info file
    info_file = output_dir / f"{args.model_name}_info.json"
    model_info = {
        'model_name': args.model_name,
        'source_checkpoint': str(args.checkpoint),
        'vocab_size': model_config.get('vocab_size', 192),
        'architecture': {
            'd_model': model_config.get('d_model', 128),
            'encoder_layers': model_config.get('encoder_layers', 8),
            'decoder_layers': model_config.get('decoder_layers', 6),
            'num_heads': model_config.get('num_heads', 4),
            'dropout': model_config.get('dropout', 0.2)
        },
        'training_info': {
            'base_epoch': checkpoint.get('epoch', 0),
            'base_val_loss': checkpoint.get('val_loss', 'N/A'),
            'base_train_loss': checkpoint.get('train_loss', 'N/A')
        },
        'files': {
            'full_model': f"{args.model_name}.pt",
            'deployment_model': f"{args.model_name}_deployment.pt"
        }
    }
    
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"""
=================================================================
FINAL MODEL CREATION COMPLETE
=================================================================
Model Name: {args.model_name}
Source: {args.checkpoint}
Output Directory: {output_dir}
Files Created:
  - {args.model_name}.pt (full model with training info)
  - {args.model_name}_deployment.pt (deployment-ready)
  - {args.model_name}_info.json (model information)
=================================================================

To use your final model:
1. For inference: Load {args.model_name}_deployment.pt
2. For further training: Load {args.model_name}.pt
3. Check {args.model_name}_info.json for details
""")

if __name__ == "__main__":
    main()