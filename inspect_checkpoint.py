#!/usr/bin/env python3
"""
Inspect PyTorch checkpoint file to understand its structure
"""
import torch
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect a PyTorch checkpoint file"""
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor {value.shape} ({value.dtype})")
            elif isinstance(value, dict):
                print(f"  {key}: Dict with {len(value)} keys")
                if len(value) < 10:  # Show dict contents if small
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: Tensor {v.shape}")
                        else:
                            print(f"    {k}: {type(v).__name__} = {v}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        
        # Check if it's a model state dict or full checkpoint
        if 'model_state_dict' in checkpoint:
            print("\n✓ Full checkpoint with training state")
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            print("\n✓ Checkpoint with state_dict")
            model_state = checkpoint['state_dict']
        else:
            print("\n? Checkpoint structure unclear")
            model_state = checkpoint
        
        # Show model architecture info
        if isinstance(model_state, dict):
            print(f"\nModel parameters ({len(model_state)} layers):")
            total_params = 0
            for name, param in model_state.items():
                if isinstance(param, torch.Tensor):
                    params = param.numel()
                    total_params += params
                    print(f"  {name}: {param.shape} ({params:,} params)")
            
            print(f"\nTotal parameters: {total_params:,}")
        
        # Show training info if available
        if 'epoch' in checkpoint:
            print(f"\nTraining info:")
            print(f"  Epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
            if 'train_loss' in checkpoint:
                print(f"  Training loss: {checkpoint['train_loss']:.4f}")
        
        return checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

if __name__ == "__main__":
    checkpoint_path = "best_model (1).pt"
    if Path(checkpoint_path).exists():
        checkpoint = inspect_checkpoint(checkpoint_path)
    else:
        print(f"Checkpoint file not found: {checkpoint_path}")