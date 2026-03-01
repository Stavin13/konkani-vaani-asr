#!/usr/bin/env python3
"""
Comprehensive search for the best checkpoint with val_loss ~2.0 and vocab_size 81
"""
import torch
from pathlib import Path
import sys

def inspect_checkpoint_comprehensive(checkpoint_path):
    """Extract comprehensive info from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic info
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', float('inf'))
        train_loss = checkpoint.get('train_loss', 'N/A')
        
        # Model architecture info
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # Try multiple ways to get vocab size
        vocab_size = 'N/A'
        d_model = 'N/A'
        
        # Check different possible keys for vocab size
        vocab_keys = [
            'ctc_head.weight', 'ctc_head.bias',
            'classifier.weight', 'classifier.bias',
            'output_layer.weight', 'output_layer.bias',
            'decoder.output_projection.weight',
            'lm_head.weight'
        ]
        
        for key in vocab_keys:
            if key in state_dict:
                vocab_size = state_dict[key].shape[0]
                break
        
        # Check different possible keys for d_model
        d_model_keys = [
            'encoder.input_proj.weight',
            'encoder.layers.0.self_attn.in_proj_weight',
            'encoder.embed_tokens.weight',
            'input_projection.weight'
        ]
        
        for key in d_model_keys:
            if key in state_dict:
                if 'in_proj_weight' in key:
                    d_model = state_dict[key].shape[1]
                else:
                    d_model = state_dict[key].shape[0]
                break
        
        # Check config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Override with config values if available
        if 'vocab_size' in model_config:
            vocab_size = model_config['vocab_size']
        if 'd_model' in model_config:
            d_model = model_config['d_model']
        
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
            'valid': True,
            'keys': list(state_dict.keys())[:10]  # First 10 keys for debugging
        }
        
    except Exception as e:
        return {
            'path': str(checkpoint_path),
            'valid': False,
            'error': str(e)
        }

def main():
    print("="*80)
    print("COMPREHENSIVE CHECKPOINT SEARCH")
    print("Looking for checkpoint with val_loss ~2.0 and vocab_size 81")
    print("="*80)
    
    # Comprehensive list of directories to search
    search_dirs = [
        'kaggle_asr_outputs/checkpoints',
        'checkpoints',
        'archives',
        'best_model (1).pt',  # Direct file
        'best_model.pt',      # Direct file
        'models',
        'outputs',
        'kaggle_downloads',
        'kaggle_outputs',
        'kaggle_best_model',
        'kaggle_new_checkpoint',
        'archives/checkpoints_backup',
        'archives/kaggle_package',
        'models/custom_konkani_model',
        '.',  # Current directory
    ]
    
    all_checkpoints = []
    
    # Search all directories
    for search_path in search_dirs:
        path = Path(search_path)
        
        if path.is_file() and path.suffix == '.pt':
            # Direct file
            all_checkpoints.append(path)
        elif path.is_dir():
            # Directory - search recursively
            for pt_file in path.rglob('*.pt'):
                if not pt_file.name.startswith('.') and 'ner' not in pt_file.name.lower():
                    all_checkpoints.append(pt_file)
    
    # Remove duplicates
    all_checkpoints = list(set(all_checkpoints))
    
    print(f"\nFound {len(all_checkpoints)} checkpoint files to analyze\n")
    
    # Inspect all checkpoints
    results = []
    for ckpt_path in all_checkpoints:
        if ckpt_path.exists():
            info = inspect_checkpoint_comprehensive(ckpt_path)
            results.append(info)
    
    # Filter valid checkpoints
    valid_results = [r for r in results if r['valid']]
    
    print(f"Valid checkpoints: {len(valid_results)}")
    
    # Sort by validation loss
    valid_results.sort(key=lambda x: x['val_loss'] if isinstance(x['val_loss'], (int, float)) else float('inf'))
    
    # Display all results with focus on val_loss < 3.0
    print("\nALL VALID CHECKPOINTS (sorted by validation loss):")
    print("-"*100)
    print(f"{'Rank':<4} {'Epoch':<6} {'Val Loss':<10} {'Vocab':<6} {'D Model':<8} {'Size MB':<8} {'Path'}")
    print("-"*100)
    
    target_checkpoints = []  # Checkpoints with val_loss < 3.0
    
    for i, r in enumerate(valid_results, 1):
        epoch = r['epoch']
        val_loss = r['val_loss']
        vocab_size = r['vocab_size']
        d_model = r['d_model']
        file_size = r['file_size_mb']
        path = Path(r['path']).name
        
        # Highlight good checkpoints
        marker = ""
        if isinstance(val_loss, (int, float)):
            if val_loss < 2.5:
                marker = "🏆"
                target_checkpoints.append(r)
            elif val_loss < 3.0:
                marker = "⭐"
                target_checkpoints.append(r)
            elif val_loss < 3.5:
                marker = "✅"
        
        print(f"{marker:<2}{i:<2} {epoch:<6} {val_loss:<10} {vocab_size:<6} {d_model:<8} {file_size:<8.1f} {path}")
    
    # Focus on target checkpoints
    if target_checkpoints:
        print(f"\n{'='*80}")
        print("TARGET CHECKPOINTS (val_loss < 3.0)")
        print(f"{'='*80}")
        
        for r in target_checkpoints:
            print(f"\n📁 {Path(r['path']).name}")
            print(f"   Path: {r['path']}")
            print(f"   Epoch: {r['epoch']}")
            print(f"   Val Loss: {r['val_loss']:.4f}")
            print(f"   Vocab Size: {r['vocab_size']}")
            print(f"   D Model: {r['d_model']}")
            print(f"   File Size: {r['file_size_mb']:.1f} MB")
            print(f"   Has Optimizer: {r['has_optimizer']}")
            print(f"   Config Available: {bool(r['config'])}")
            
            # Check if this matches what you're looking for
            val_loss = r['val_loss']
            vocab_size = r['vocab_size']
            
            if isinstance(val_loss, (int, float)) and isinstance(vocab_size, int):
                if 1.9 <= val_loss <= 2.1 and vocab_size == 81:
                    print(f"   🎯 PERFECT MATCH! Val loss ~2.0 and vocab_size 81")
                elif vocab_size == 81:
                    print(f"   ✅ VOCAB MATCH! Has vocab_size 81")
                elif 1.9 <= val_loss <= 2.1:
                    print(f"   ✅ LOSS MATCH! Has val_loss ~2.0")
    
    # Find the absolute best
    if valid_results:
        best = valid_results[0]
        print(f"\n{'='*80}")
        print("ABSOLUTE BEST CHECKPOINT")
        print(f"{'='*80}")
        print(f"Path: {best['path']}")
        print(f"Epoch: {best['epoch']}")
        print(f"Validation Loss: {best['val_loss']:.4f}")
        print(f"Vocabulary Size: {best['vocab_size']}")
        print(f"Model Dimension: {best['d_model']}")
        
        # Command to use this checkpoint
        print(f"\n🚀 TO USE THIS CHECKPOINT:")
        print(f"python scripts/create_final_model.py \\")
        print(f"    --checkpoint \"{best['path']}\" \\")
        print(f"    --output_dir \"final_models\" \\")
        print(f"    --model_name \"konkanivani_best_val{best['val_loss']:.3f}\" \\")
        print(f"    --skip_training")
        
        return best['path']
    else:
        print("\n❌ No valid checkpoints found!")
        return None

if __name__ == '__main__':
    main()