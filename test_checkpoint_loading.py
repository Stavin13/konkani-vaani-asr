#!/usr/bin/env python3
"""
Test script to verify checkpoint can be loaded and model works
"""
import torch
import json
import sys
from pathlib import Path

def test_checkpoint_loading(checkpoint_path):
    """Test loading the checkpoint"""
    print(f"Testing checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Checkpoint loaded successfully")
        
        # Check contents
        print(f"✓ Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"✓ Validation loss: {checkpoint.get('val_loss', 'Unknown')}")
        print(f"✓ Training loss: {checkpoint.get('train_loss', 'Unknown')}")
        
        # Check model state
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"✓ Model state dict with {len(model_state)} parameters")
            
            # Count total parameters
            total_params = 0
            for name, param in model_state.items():
                if isinstance(param, torch.Tensor):
                    total_params += param.numel()
            
            print(f"✓ Total parameters: {total_params:,}")
        
        # Check optimizer state
        if 'optimizer_state_dict' in checkpoint:
            print("✓ Optimizer state available")
        
        # Check scheduler state
        if 'scheduler_state_dict' in checkpoint:
            print("✓ Scheduler state available")
        
        # Check config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("✓ Training config available:")
            if 'model' in config:
                model_config = config['model']
                print(f"  - Vocab size: {model_config.get('vocab_size', 'Unknown')}")
                print(f"  - Model dim: {model_config.get('d_model', 'Unknown')}")
                print(f"  - Encoder layers: {model_config.get('encoder_layers', 'Unknown')}")
                print(f"  - Decoder layers: {model_config.get('decoder_layers', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False

def test_model_creation():
    """Test if we can create the model"""
    print("\nTesting model creation...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Try to import the model
        from models.konkanivani_asr import create_konkanivani_model
        
        # Create a test model
        model = create_konkanivani_model(
            vocab_size=81,
            d_model=128,
            encoder_layers=8,
            decoder_layers=6,
            num_heads=4,
            dropout=0.3,
            conv_kernel_size=31
        )
        
        print("✓ Model created successfully")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 100
        feature_dim = 80
        
        # Dummy input
        audio_features = torch.randn(batch_size, seq_len, feature_dim)
        text_input = torch.randint(0, 81, (batch_size, 20))
        
        model.eval()
        with torch.no_grad():
            outputs = model(audio_features, text_input)
        
        print("✓ Forward pass successful")
        print(f"✓ Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Single tensor'}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import model: {e}")
        print("Make sure models/konkanivani_asr.py is available")
        return False
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

def test_data_loading():
    """Test if we can load the data files"""
    print("\nTesting data file access...")
    
    # Check for common data files
    data_files = [
        "konkani-10k/train_manifest.json",
        "konkani-10k/val_manifest.json", 
        "konkani-10k/vocab.json"
    ]
    
    all_found = True
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✓ Found: {file_path}")
            
            # Try to load and check format
            try:
                if file_path.endswith('vocab.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        vocab = json.load(f)
                    print(f"  - Vocab size: {len(vocab.get('char2idx', {}))}")
                else:
                    # Count lines in manifest
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = sum(1 for line in f if line.strip())
                    print(f"  - Samples: {lines}")
            except Exception as e:
                print(f"  ⚠️  Could not read file: {e}")
        else:
            print(f"✗ Missing: {file_path}")
            all_found = False
    
    return all_found

def main():
    print("=== Checkpoint and Model Testing ===\n")
    
    # Test checkpoint loading
    checkpoint_path = "best_model (1).pt"
    if Path(checkpoint_path).exists():
        checkpoint_ok = test_checkpoint_loading(checkpoint_path)
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        checkpoint_ok = False
    
    # Test model creation
    model_ok = test_model_creation()
    
    # Test data loading
    data_ok = test_data_loading()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Checkpoint loading: {'✓ PASS' if checkpoint_ok else '✗ FAIL'}")
    print(f"Model creation: {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Data files: {'✓ PASS' if data_ok else '✗ FAIL'}")
    
    if checkpoint_ok and model_ok and data_ok:
        print("\n🎉 All tests passed! You're ready to fine-tune.")
        print("\nNext steps:")
        print("1. Run: python fine_tune_from_checkpoint.py --help")
        print("2. Start fine-tuning with your desired parameters")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before fine-tuning.")
        
        if not checkpoint_ok:
            print("- Check that 'best_model (1).pt' exists and is valid")
        if not model_ok:
            print("- Make sure models/konkanivani_asr.py is available")
        if not data_ok:
            print("- Check that data files exist and are accessible")

if __name__ == "__main__":
    main()