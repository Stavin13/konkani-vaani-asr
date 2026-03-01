#!/usr/bin/env python3
"""
Quick accuracy test for top checkpoints
"""
import torch
import json
from pathlib import Path
import sys

def inspect_checkpoint_detailed(checkpoint_path):
    """Get detailed info from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic info
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', float('inf'))
        train_loss = checkpoint.get('train_loss', 'N/A')
        
        # Model architecture info
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
        
        # Try to get vocab size from model
        vocab_size = 'N/A'
        d_model = 'N/A'
        
        if 'ctc_head.weight' in state_dict:
            vocab_size = state_dict['ctc_head.weight'].shape[0]
        elif 'classifier.weight' in state_dict:
            vocab_size = state_dict['classifier.weight'].shape[0]
        
        if 'encoder.input_proj.weight' in state_dict:
            d_model = state_dict['encoder.input_proj.weight'].shape[0]
        elif 'encoder.layers.0.self_attn.in_proj_weight' in state_dict:
            d_model = state_dict['encoder.layers.0.self_attn.in_proj_weight'].shape[1]
        
        # Check for config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        return {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'vocab_size': vocab_size,
            'd_model': d_model,
            'config': model_config,
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
            'has_scheduler': 'scheduler_state_dict' in checkpoint,
            'file_size_mb': checkpoint_path.stat().st_size / (1024*1024),
            'valid': True
        }
        
    except Exception as e:
        return {
            'path': str(checkpoint_path),
            'valid': False,
            'error': str(e)
        }

def main():
    print("="*80)
    print("DETAILED CHECKPOINT ANALYSIS")
    print("="*80)
    
    # Top checkpoints from previous analysis
    top_checkpoints = [
        'kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt',
        'kaggle_asr_outputs/checkpoints/checkpoint_epoch_19.pt', 
        'kaggle_asr_outputs/checkpoints/checkpoint_epoch_14.pt',
        'kaggle_asr_outputs/checkpoints/checkpoint_epoch_35.pt',
        'kaggle_asr_outputs/checkpoints/checkpoint_epoch_25.pt'
    ]
    
    results = []
    
    for checkpoint_path in top_checkpoints:
        path = Path(checkpoint_path)
        if path.exists():
            print(f"\nAnalyzing: {path.name}")
            info = inspect_checkpoint_detailed(path)
            results.append(info)
            
            if info['valid']:
                print(f"  ✅ Epoch: {info['epoch']}")
                print(f"     Val Loss: {info['val_loss']:.4f}")
                print(f"     Vocab Size: {info['vocab_size']}")
                print(f"     D Model: {info['d_model']}")
                print(f"     File Size: {info['file_size_mb']:.1f} MB")
                print(f"     Has Optimizer: {info['has_optimizer']}")
                print(f"     Config: {bool(info['config'])}")
            else:
                print(f"  ❌ Error: {info['error']}")
        else:
            print(f"\n❌ Not found: {checkpoint_path}")
    
    # Find best checkpoint
    valid_results = [r for r in results if r['valid']]
    if valid_results:
        best = min(valid_results, key=lambda x: x['val_loss'])
        
        print("\n" + "="*80)
        print("RECOMMENDED CHECKPOINT FOR FINAL MODEL")
        print("="*80)
        print(f"Path: {best['path']}")
        print(f"Epoch: {best['epoch']}")
        print(f"Validation Loss: {best['val_loss']:.4f}")
        print(f"Vocabulary Size: {best['vocab_size']}")
        print(f"Model Dimension: {best['d_model']}")
        print(f"File Size: {best['file_size_mb']:.1f} MB")
        
        # Check if vocab size matches current dataset
        expected_vocab_size = 192  # From mega dataset
        actual_vocab_size = best['vocab_size']
        
        if actual_vocab_size != expected_vocab_size:
            print(f"\n⚠️  VOCABULARY MISMATCH DETECTED!")
            print(f"   Model vocab size: {actual_vocab_size}")
            print(f"   Dataset vocab size: {expected_vocab_size}")
            print(f"   This may cause poor performance!")
        else:
            print(f"\n✅ Vocabulary size matches dataset ({expected_vocab_size})")
        
        # Create command to use this checkpoint
        print(f"\n🚀 TO CREATE FINAL MODEL:")
        print(f"python scripts/create_final_model.py \\")
        print(f"    --checkpoint \"{best['path']}\" \\")
        print(f"    --output_dir \"final_models\" \\")
        print(f"    --model_name \"konkanivani_best_epoch{best['epoch']}\" \\")
        print(f"    --skip_training")
        
        return best['path']
    else:
        print("\n❌ No valid checkpoints found!")
        return None

if __name__ == '__main__':
    main()