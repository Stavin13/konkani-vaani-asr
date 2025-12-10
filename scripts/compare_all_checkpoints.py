#!/usr/bin/env python3
"""
Compare all ASR checkpoints and find the best one
"""
import torch
from pathlib import Path
import sys

def inspect_checkpoint(checkpoint_path):
    """Extract key info from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'train_loss': checkpoint.get('train_loss', 'N/A'),
            'valid': True
        }
        
        # Check if it has model weights
        if 'model_state_dict' not in checkpoint:
            info['valid'] = False
            info['error'] = 'No model_state_dict'
        
        return info
    except Exception as e:
        return {
            'path': str(checkpoint_path),
            'valid': False,
            'error': str(e)
        }

def main():
    # Find all ASR checkpoints (exclude NER models)
    checkpoint_dirs = [
        'kaggle_asr_outputs/checkpoints',
        'checkpoints',
        'archives/checkpoints_backup',
        'archives',
        'models/custom_konkani_model',
        'archives/kaggle_package/models/custom_konkani_model'
    ]
    
    all_checkpoints = []
    for dir_path in checkpoint_dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            # Find .pt files, exclude NER and hidden files
            for pt_file in dir_path.glob('*.pt'):
                if not pt_file.name.startswith('.') and 'ner' not in pt_file.name.lower():
                    all_checkpoints.append(pt_file)
    
    print("="*80)
    print("COMPARING ALL ASR CHECKPOINTS")
    print("="*80)
    print(f"\nFound {len(all_checkpoints)} checkpoint files\n")
    
    # Inspect all checkpoints
    results = []
    for ckpt_path in all_checkpoints:
        info = inspect_checkpoint(ckpt_path)
        results.append(info)
    
    # Filter valid checkpoints
    valid_results = [r for r in results if r['valid']]
    invalid_results = [r for r in results if not r['valid']]
    
    # Sort by validation loss
    valid_results.sort(key=lambda x: x['val_loss'])
    
    # Display results
    print("VALID CHECKPOINTS (sorted by validation loss):")
    print("-"*80)
    for i, r in enumerate(valid_results[:15], 1):  # Show top 15
        epoch = r['epoch']
        val_loss = r['val_loss']
        train_loss = r.get('train_loss', 'N/A')
        path = r['path']
        
        marker = "⭐ BEST" if i == 1 else f"  #{i}"
        print(f"{marker:8} | Epoch {epoch:3} | Val Loss: {val_loss:8.4f} | Train Loss: {train_loss} | {path}")
    
    if len(valid_results) > 15:
        print(f"\n... and {len(valid_results) - 15} more checkpoints")
    
    if invalid_results:
        print(f"\n\nINVALID/CORRUPTED CHECKPOINTS: {len(invalid_results)}")
        print("-"*80)
        for r in invalid_results[:5]:
            print(f"✗ {r['path']}")
            print(f"  Error: {r.get('error', 'Unknown')}")
    
    # Show best checkpoint
    if valid_results:
        best = valid_results[0]
        print("\n" + "="*80)
        print("BEST CHECKPOINT")
        print("="*80)
        print(f"Path: {best['path']}")
        print(f"Epoch: {best['epoch']}")
        print(f"Validation Loss: {best['val_loss']:.4f}")
        if best.get('train_loss') != 'N/A':
            print(f"Training Loss: {best['train_loss']}")
        
        return best['path']
    else:
        print("\n✗ No valid checkpoints found!")
        return None

if __name__ == '__main__':
    best_path = main()
    if best_path:
        sys.exit(0)
    else:
        sys.exit(1)
