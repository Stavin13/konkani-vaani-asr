#!/usr/bin/env python3
"""
Investigate Config vs Model Mismatch
===================================
The YAML says vocab_size=200, but model has vocab_size=81
"""
import torch
import yaml
from pathlib import Path
import json

def investigate_config_mismatch():
    """Investigate why config and model don't match"""
    
    print("="*80)
    print("CONFIG vs MODEL MISMATCH INVESTIGATION")
    print("="*80)
    
    # 1. Check YAML config
    print("\n1. YAML CONFIGURATION:")
    print("-" * 40)
    
    yaml_path = Path('config/training_config_from_checkpoint15.yaml')
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        yaml_vocab_size = config['model']['vocab_size']
        print(f"YAML vocab_size: {yaml_vocab_size}")
        print(f"YAML d_model: {config['model']['d_model']}")
        print(f"YAML encoder_layers: {config['model']['encoder_layers']}")
    else:
        print("❌ YAML config not found")
        return
    
    # 2. Check actual model
    print("\n2. ACTUAL MODEL IN CHECKPOINT:")
    print("-" * 40)
    
    checkpoint_path = Path('/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', {})
        
        model_vocab_size = state_dict['ctc_head.weight'].shape[0]
        model_d_model = state_dict['encoder.input_proj.weight'].shape[0]
        
        print(f"Model vocab_size: {model_vocab_size}")
        print(f"Model d_model: {model_d_model}")
        
        # Check if config was stored in checkpoint
        stored_config = checkpoint.get('config', {})
        print(f"Config in checkpoint: {stored_config}")
        
    else:
        print("❌ Checkpoint not found")
        return
    
    # 3. Check what data actually needs
    print("\n3. WHAT DATA ACTUALLY NEEDS:")
    print("-" * 40)
    
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    if manifest_path.exists():
        # Count unique characters
        chars = set()
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                text = sample.get('text', '')
                chars.update(text)
        
        data_vocab_size = len(chars) + 2  # +2 for <blank> and <unk>
        print(f"Data needs vocab_size: {data_vocab_size}")
        print(f"Unique characters in data: {len(chars)}")
    else:
        print("❌ Training manifest not found")
        return
    
    # 4. Analyze the mismatch
    print("\n4. MISMATCH ANALYSIS:")
    print("-" * 40)
    
    print(f"YAML config:     vocab_size = {yaml_vocab_size}")
    print(f"Actual model:    vocab_size = {model_vocab_size}")
    print(f"Data requires:   vocab_size = {data_vocab_size}")
    
    print(f"\n🔍 FINDINGS:")
    
    if yaml_vocab_size != model_vocab_size:
        print(f"❌ CRITICAL: YAML config was IGNORED during training!")
        print(f"   → Model was trained with vocab_size={model_vocab_size}")
        print(f"   → But YAML specifies vocab_size={yaml_vocab_size}")
        print(f"   → This suggests the training script didn't use the YAML config")
    
    if model_vocab_size < data_vocab_size:
        print(f"❌ CRITICAL: Model is too small for the data!")
        print(f"   → {data_vocab_size - model_vocab_size} characters will be mapped to <unk>")
    
    if yaml_vocab_size >= data_vocab_size:
        print(f"✅ YAML config vocab_size ({yaml_vocab_size}) is sufficient for data ({data_vocab_size})")
        print(f"   → The YAML config was actually correct!")
        print(f"   → The problem is that training didn't use the YAML config")
    
    # 5. Find the root cause
    print(f"\n5. ROOT CAUSE ANALYSIS:")
    print("-" * 40)
    
    print("Possible reasons for config mismatch:")
    print("1. Training script hardcoded vocab_size=81")
    print("2. Training script used a different config file")
    print("3. Training script didn't load the YAML config properly")
    print("4. Model was initialized before loading config")
    print("5. Kaggle training used different parameters")
    
    # 6. Check training scripts
    print(f"\n6. CHECKING TRAINING SCRIPTS:")
    print("-" * 40)
    
    training_scripts = [
        'training_scripts/train_konkanivani_asr.py',
        'notebooks/KAGGLE_RETRAIN_WITH_TESTING.ipynb',
        'notebooks/KAGGLE_TRAINING_OPTIMIZED.ipynb'
    ]
    
    for script_path in training_scripts:
        if Path(script_path).exists():
            print(f"Found: {script_path}")
            # Could analyze the script to see how vocab_size is set
        else:
            print(f"Missing: {script_path}")
    
    # 7. Provide solution
    print(f"\n7. SOLUTION:")
    print("-" * 40)
    
    print("The YAML config is actually CORRECT (vocab_size=200 > required 193)")
    print("The problem is that training didn't use this config.")
    print()
    print("To fix this:")
    print("1. Retrain using the YAML config properly")
    print("2. Ensure training script loads vocab_size from YAML")
    print("3. Verify model initialization uses config values")
    print()
    print("Quick fix command:")
    print(f"# Create model with YAML config vocab_size")
    print(f"python scripts/create_model_from_yaml_config.py")

def create_model_from_yaml_config():
    """Create a model using the correct YAML configuration"""
    
    print("\n" + "="*80)
    print("CREATING MODEL FROM YAML CONFIG")
    print("="*80)
    
    # Load YAML config
    yaml_path = Path('config/training_config_from_checkpoint15.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    print("Creating model with YAML config:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Import and create model
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.konkanivani_asr import KonkaniVaniASR
    
    model = KonkaniVaniASR(
        vocab_size=model_config['vocab_size'],  # 200 from YAML
        input_dim=model_config['input_dim'],
        d_model=model_config['d_model'],
        encoder_layers=model_config['encoder_layers'],
        decoder_layers=model_config['decoder_layers'],
        num_heads=model_config['num_heads'],
        conv_kernel_size=model_config['conv_kernel_size'],
        dropout=model_config['dropout']
    )
    
    print(f"\n✅ Model created successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {model_config['vocab_size']}")
    
    # Save initial checkpoint
    checkpoint_dir = Path('checkpoints/yaml_config_model')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': model_config['vocab_size']
    }
    
    checkpoint_path = checkpoint_dir / 'initial_model_yaml_config.pt'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Initial checkpoint saved: {checkpoint_path}")
    print(f"\nThis model can now be trained with the correct vocab_size=200")
    print(f"which is sufficient for your data (needs 193)")

if __name__ == '__main__':
    investigate_config_mismatch()
    create_model_from_yaml_config()