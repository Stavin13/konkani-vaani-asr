#!/usr/bin/env python3
"""
Setup script for RTX 3060 training environment
Checks model, data, and prepares for training
"""
import torch
import json
import os
from pathlib import Path
import sys

def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please install PyTorch with CUDA support.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ VRAM: {total_memory:.1f}GB")
    
    if "3060" in gpu_name:
        print("✓ RTX 3060 detected - optimizations will be applied")
    elif total_memory < 8:
        print("⚠️  Low VRAM detected - using memory optimizations")
    
    return True

def check_model(model_path="best_model (1).pt"):
    """Check if model file exists and inspect it"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✓ Model loaded: {model_path}")
        
        # Check model info
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  - Config found: {config.keys()}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  - Parameters: {len(state_dict)} layers")
            
            # Estimate model size
            total_params = 0
            for param in state_dict.values():
                if hasattr(param, 'numel'):
                    total_params += param.numel()
            print(f"  - Total parameters: {total_params:,}")
        
        if 'epoch' in checkpoint:
            print(f"  - Trained epochs: {checkpoint['epoch']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def check_data():
    """Check if data files exist"""
    data_files = [
        "data/konkani-10k/train_manifest.json",
        "data/konkani-10k/val_manifest.json", 
        "data/konkani-10k/vocab.json"
    ]
    
    missing_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
            
            # Check file content
            if file_path.endswith('.json') and 'manifest' in file_path:
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    print(f"  - Samples: {len(lines)}")
                except:
                    print(f"  - Could not read sample count")
                    
        else:
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
    
    if missing_files:
        print("\n⚠️  Some data files are missing. Checking alternatives...")
        
        # Check for alternative data locations
        alternative_locations = [
            "data/konkani-full/",
            "data/nllb_finetuning/",
            "data/konkani-raw-corpus/manifests/"
        ]
        
        for location in alternative_locations:
            if os.path.exists(location):
                files = os.listdir(location)
                json_files = [f for f in files if f.endswith('.json')]
                if json_files:
                    print(f"✓ Alternative data found in: {location}")
                    print(f"  Files: {json_files}")
    
    return len(missing_files) == 0

def check_dependencies():
    """Check required dependencies"""
    required_packages = [
        'torch', 'torchaudio', 'librosa', 'numpy', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_training_config():
    """Create optimized training configuration"""
    config = {
        "model": {
            "vocab_size": 81,
            "input_dim": 80,
            "d_model": 128,
            "encoder_layers": 6,
            "decoder_layers": 4,
            "num_heads": 4,
            "dropout": 0.1,
            "conv_kernel_size": 15
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.00003,
            "epochs": 10,
            "max_audio_length": 128000,
            "use_mixed_precision": True
        },
        "rtx3060_optimizations": {
            "memory_fraction": 0.95,
            "gradient_checkpointing": True,
            "num_workers": 1,
            "pin_memory": False
        }
    }
    
    with open('rtx3060_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created rtx3060_config.json")
    return config

def main():
    print("=================================================================")
    print("RTX 3060 KONKANI ASR FINE-TUNING SETUP")
    print("=================================================================\n")
    
    # Check GPU
    print("1. Checking GPU...")
    if not check_gpu():
        return
    print()
    
    # Check dependencies
    print("2. Checking dependencies...")
    if not check_dependencies():
        return
    print()
    
    # Check model
    print("3. Checking model...")
    if not check_model():
        return
    print()
    
    # Check data
    print("4. Checking data...")
    check_data()
    print()
    
    # Create config
    print("5. Creating training configuration...")
    config = create_training_config()
    print()
    
    print("=================================================================")
    print("SETUP COMPLETE!")
    print("=================================================================")
    print("Ready to start training. Run:")
    print("python finetune_rtx3060.py")
    print()
    print("Optional parameters:")
    print("--epochs 15              # More epochs")
    print("--learning_rate 0.00005  # Higher learning rate")
    print("--checkpoint best_model.pt  # Different model file")
    print("=================================================================")

if __name__ == "__main__":
    main()