#!/usr/bin/env python3
"""
Inspect checkpoint to understand what the model learned
"""
import torch
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint contents"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*70)
    print("CHECKPOINT CONTENTS")
    print("="*70)
    
    # Basic info
    print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
    
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    if 'train_loss' in checkpoint:
        print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    
    if 'val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    
    if 'train_wer' in checkpoint:
        print(f"Train WER: {checkpoint.get('train_wer', 'N/A')}")
    
    if 'val_wer' in checkpoint:
        print(f"Val WER: {checkpoint.get('val_wer', 'N/A')}")
    
    # Check model state
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nModel parameters: {len(state_dict)} tensors")
        
        # Check if weights are actually trained (not all zeros/random)
        total_params = 0
        zero_params = 0
        
        for name, param in state_dict.items():
            total_params += param.numel()
            if torch.all(param == 0):
                zero_params += param.numel()
        
        print(f"Total parameters: {total_params:,}")
        print(f"Zero parameters: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
        
        # Sample some weights
        print("\nSample weight statistics:")
        sample_keys = list(state_dict.keys())[:5]
        for key in sample_keys:
            param = state_dict[key]
            print(f"  {key}:")
            print(f"    Shape: {param.shape}")
            print(f"    Mean: {param.mean().item():.6f}")
            print(f"    Std: {param.std().item():.6f}")
            print(f"    Min: {param.min().item():.6f}")
            print(f"    Max: {param.max().item():.6f}")
    
    # Check optimizer state
    if 'optimizer_state_dict' in checkpoint:
        print("\n✓ Optimizer state present")
    
    # Check if there's training history
    if 'history' in checkpoint:
        history = checkpoint['history']
        print(f"\nTraining history available: {len(history)} epochs")
        if history:
            print(f"  First epoch loss: {history[0].get('val_loss', 'N/A')}")
            print(f"  Last epoch loss: {history[-1].get('val_loss', 'N/A')}")
    
    print("\n" + "="*70)


def compare_checkpoints(checkpoint_paths):
    """Compare multiple checkpoints"""
    print("\n" + "="*70)
    print("CHECKPOINT COMPARISON")
    print("="*70 + "\n")
    
    results = []
    for path in checkpoint_paths:
        if not Path(path).exists():
            print(f"⚠ Skipping {path} (not found)")
            continue
        
        checkpoint = torch.load(path, map_location='cpu')
        results.append({
            'path': path,
            'epoch': checkpoint.get('epoch', 'N/A'),
            'train_loss': checkpoint.get('train_loss', 'N/A'),
            'val_loss': checkpoint.get('val_loss', 'N/A'),
            'train_wer': checkpoint.get('train_wer', 'N/A'),
            'val_wer': checkpoint.get('val_wer', 'N/A')
        })
    
    # Print comparison table
    print(f"{'Checkpoint':<30} {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val WER':<12}")
    print("-" * 80)
    for r in results:
        epoch = str(r['epoch'])
        train_loss = f"{r['train_loss']:.4f}" if isinstance(r['train_loss'], float) else str(r['train_loss'])
        val_loss = f"{r['val_loss']:.4f}" if isinstance(r['val_loss'], float) else str(r['val_loss'])
        val_wer = f"{r['val_wer']:.2f}%" if isinstance(r['val_wer'], float) else str(r['val_wer'])
        
        print(f"{Path(r['path']).name:<30} {epoch:<8} {train_loss:<12} {val_loss:<12} {val_wer:<12}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='kaggle_asr_outputs/checkpoints/best_model.pt')
    parser.add_argument('--compare', nargs='+', help='Compare multiple checkpoints')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_checkpoints(args.compare)
    else:
        inspect_checkpoint(args.checkpoint)
