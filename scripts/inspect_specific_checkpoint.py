#!/usr/bin/env python3
"""
Inspect the specific checkpoint at /Volumes/data&proj/konkani/best_model (1).pt
"""
import torch
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint structure and contents"""
    
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
    
    # Check if file exists
    if not Path(checkpoint_path).exists():
        print(f"❌ File not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    # Print checkpoint structure
    print(f"\n📊 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Inspect model state dict
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print(f"\n🏗️  Model state dict keys (first 10):")
        for i, key in enumerate(list(model_state.keys())[:10]):
            shape = model_state[key].shape if hasattr(model_state[key], 'shape') else 'N/A'
            print(f"  {i+1}. {key}: {shape}")
        
        if len(model_state.keys()) > 10:
            print(f"  ... and {len(model_state.keys()) - 10} more keys")
        
        # Check for DataParallel wrapper
        has_module_prefix = any(key.startswith('module.') for key in model_state.keys())
        print(f"\n🔧 Has 'module.' prefix (DataParallel): {has_module_prefix}")
        
        # Infer model architecture
        try:
            # Get vocab size
            vocab_size = None
            if 'vocab_size' in checkpoint:
                vocab_size = checkpoint['vocab_size']
            else:
                ctc_key = 'ctc_head.weight' if 'ctc_head.weight' in model_state else 'module.ctc_head.weight'
                if ctc_key in model_state:
                    vocab_size = model_state[ctc_key].shape[0]
            
            # Get d_model
            d_model = None
            encoder_key = 'encoder.input_proj.weight' if 'encoder.input_proj.weight' in model_state else 'module.encoder.input_proj.weight'
            if encoder_key in model_state:
                d_model = model_state[encoder_key].shape[0]
            
            # Count layers
            encoder_layers = 0
            decoder_layers = 0
            
            for key in model_state.keys():
                if 'encoder.layers.' in key and '.ff1.0.weight' in key:
                    layer_num = int(key.split('.')[2] if not key.startswith('module.') else key.split('.')[3])
                    encoder_layers = max(encoder_layers, layer_num + 1)
                
                if 'decoder.decoder.layers.' in key and '.linear1.weight' in key:
                    layer_num = int(key.split('.')[3] if not key.startswith('module.') else key.split('.')[4])
                    decoder_layers = max(decoder_layers, layer_num + 1)
            
            print(f"\n📐 Inferred model architecture:")
            print(f"  - vocab_size: {vocab_size}")
            print(f"  - d_model: {d_model}")
            print(f"  - encoder_layers: {encoder_layers}")
            print(f"  - decoder_layers: {decoder_layers}")
            
        except Exception as e:
            print(f"⚠️  Could not infer architecture: {e}")
    
    # Check training info
    epoch = checkpoint.get('epoch', 'Not found')
    print(f"\n📈 Training info:")
    print(f"  - Last epoch: {epoch}")
    print(f"  - Has optimizer state: {'optimizer_state_dict' in checkpoint}")
    print(f"  - Has scheduler state: {'scheduler_state_dict' in checkpoint}")
    
    # Check vocabulary info
    has_vocab = 'idx_to_char' in checkpoint and 'char_to_idx' in checkpoint
    print(f"  - Has vocabulary: {has_vocab}")
    if has_vocab:
        print(f"  - Vocab size from dict: {len(checkpoint['char_to_idx'])}")
    
    # Check other keys
    other_keys = [k for k in checkpoint.keys() if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'idx_to_char', 'char_to_idx', 'vocab_size']]
    if other_keys:
        print(f"\n🔍 Other keys: {other_keys}")
    
    return checkpoint

def main():
    checkpoint_path = "/Volumes/data&proj/konkani/best_model (1).pt"
    checkpoint = inspect_checkpoint(checkpoint_path)
    
    if checkpoint:
        print(f"\n✅ Checkpoint inspection complete!")
        print(f"📝 This checkpoint can be used for resuming training")

if __name__ == "__main__":
    main()