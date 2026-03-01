#!/usr/bin/env python3
"""
Quick Start Script for RTX 3060 Konkani ASR Fine-tuning
Automatically detects best data sources and starts training
"""
import os
import sys
import json
import subprocess
from pathlib import Path

def find_best_data_source():
    """Find the best available data source"""
    data_sources = [
        {
            'name': 'konkani-10k',
            'train': 'data/konkani-10k/train_manifest.json',
            'val': 'data/konkani-10k/val_manifest.json',
            'vocab': 'data/konkani-10k/vocab.json',
            'priority': 1
        },
        {
            'name': 'konkani-full',
            'train': 'data/konkani-full/train.json',
            'val': 'data/konkani-full/val.json',
            'vocab': 'data/konkani-10k/vocab.json',  # Use 10k vocab
            'priority': 2
        },
        {
            'name': 'nllb-finetuning',
            'train': 'data/nllb_finetuning/train.json',
            'val': 'data/nllb_finetuning/val.json',
            'vocab': 'data/konkani-10k/vocab.json',
            'priority': 3
        },
        {
            'name': 'raw-corpus',
            'train': 'data/konkani-raw-corpus/manifests/train.json',
            'val': 'data/konkani-raw-corpus/manifests/val.json',
            'vocab': 'data/konkani-10k/vocab.json',
            'priority': 4
        }
    ]
    
    available_sources = []
    for source in data_sources:
        if all(os.path.exists(f) for f in [source['train'], source['val'], source['vocab']]):
            available_sources.append(source)
            print(f"✓ Found complete dataset: {source['name']}")
        else:
            missing = [f for f in [source['train'], source['val'], source['vocab']] if not os.path.exists(f)]
            print(f"❌ Incomplete dataset {source['name']}: missing {missing}")
    
    if not available_sources:
        return None
    
    # Return highest priority (lowest number) available source
    best_source = min(available_sources, key=lambda x: x['priority'])
    print(f"\n🎯 Selected dataset: {best_source['name']}")
    return best_source

def find_model_file():
    """Find the best available model file"""
    model_candidates = [
        'best_model (1).pt',
        'best_model.pt',
        'checkpoints/best_model.pt',
        'models/best_model.pt'
    ]
    
    for model_path in model_candidates:
        if os.path.exists(model_path):
            print(f"✓ Found model: {model_path}")
            return model_path
    
    print("❌ No model file found!")
    print("Expected locations:")
    for path in model_candidates:
        print(f"  - {path}")
    return None

def check_system_requirements():
    """Check if system meets requirements"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ VRAM: {vram_gb:.1f}GB")
        
        if vram_gb < 4:
            print("⚠️  Warning: Less than 4GB VRAM detected")
            print("   Training may fail or be very slow")
        
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def create_optimized_command(data_source, model_file):
    """Create optimized training command"""
    base_cmd = [
        sys.executable, 'finetune_rtx3060.py',
        '--checkpoint', model_file,
        '--train_manifest', data_source['train'],
        '--val_manifest', data_source['val'],
        '--vocab_file', data_source['vocab']
    ]
    
    # Add RTX 3060 specific optimizations
    rtx3060_args = [
        '--epochs', '10',
        '--learning_rate', '0.00003',
        '--output_dir', 'rtx3060_output'
    ]
    
    return base_cmd + rtx3060_args

def main():
    print("=================================================================")
    print("🚀 QUICK START - RTX 3060 KONKANI ASR FINE-TUNING")
    print("=================================================================\n")
    
    # Check system
    print("1. Checking system requirements...")
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please install PyTorch with CUDA.")
        return
    print()
    
    # Find model
    print("2. Looking for model file...")
    model_file = find_model_file()
    if not model_file:
        return
    print()
    
    # Find data
    print("3. Looking for training data...")
    data_source = find_best_data_source()
    if not data_source:
        print("\n❌ No complete dataset found!")
        print("Please ensure you have train/val manifests and vocab.json")
        return
    print()
    
    # Create command
    print("4. Preparing training command...")
    cmd = create_optimized_command(data_source, model_file)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Confirm and start
    print("=================================================================")
    print("🎯 READY TO START TRAINING")
    print("=================================================================")
    print(f"Model: {model_file}")
    print(f"Dataset: {data_source['name']}")
    print(f"Train samples: {data_source['train']}")
    print(f"Val samples: {data_source['val']}")
    print(f"Vocabulary: {data_source['vocab']}")
    print()
    
    response = input("Start training now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting training...")
        print("=================================================================")
        
        try:
            # Run the training
            result = subprocess.run(cmd, check=True)
            print("\n✅ Training completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error code: {e.returncode}")
            print("Check the error messages above for details.")
            
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            
    else:
        print("\n📝 To start training manually, run:")
        print(f"python {' '.join(cmd[1:])}")
        print("\n💡 Or run the setup script first:")
        print("python setup_rtx3060_training.py")

if __name__ == "__main__":
    main()