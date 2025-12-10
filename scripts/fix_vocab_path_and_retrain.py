#!/usr/bin/env python3
"""
Fix Vocabulary Path and Create Correct Model
===========================================
Use the correct vocab.json (200 chars) instead of konkani-10k/vocab.json (81 chars)
"""
import torch
import json
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR

def fix_vocab_path_and_retrain():
    """Fix the vocabulary path issue and create correct model"""
    
    print("="*80)
    print("FIXING VOCABULARY PATH ISSUE")
    print("="*80)
    
    # 1. Compare the two vocab files
    print("\n1. COMPARING VOCABULARY FILES:")
    print("-" * 50)
    
    main_vocab_path = Path('data/vocab.json')
    wrong_vocab_path = Path('data/konkani-10k/vocab.json')
    
    # Load both vocab files
    with open(main_vocab_path, 'r') as f:
        main_vocab = json.load(f)['char2idx']
    
    with open(wrong_vocab_path, 'r') as f:
        wrong_vocab = json.load(f)['char2idx']
    
    print(f"Main vocab (data/vocab.json): {len(main_vocab)} characters")
    print(f"Wrong vocab (data/konkani-10k/vocab.json): {len(wrong_vocab)} characters")
    
    print(f"\n🎯 ISSUE IDENTIFIED:")
    print(f"   Training used: {wrong_vocab_path} ({len(wrong_vocab)} chars)")
    print(f"   Should use:    {main_vocab_path} ({len(main_vocab)} chars)")
    
    # 2. Load YAML config
    print(f"\n2. LOADING YAML CONFIGURATION:")
    print("-" * 50)
    
    yaml_path = Path('config/training_config_from_checkpoint15.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    yaml_vocab_size = model_config['vocab_size']
    
    print(f"YAML vocab_size: {yaml_vocab_size}")
    print(f"Main vocab size: {len(main_vocab)}")
    
    if yaml_vocab_size == len(main_vocab):
        print("✅ YAML config matches main vocab file!")
    else:
        print("⚠️  YAML config doesn't match main vocab file")
    
    # 3. Create model with correct vocabulary
    print(f"\n3. CREATING MODEL WITH CORRECT VOCABULARY:")
    print("-" * 50)
    
    # Use the main vocab file size
    correct_vocab_size = len(main_vocab)
    
    model = KonkaniVaniASR(
        vocab_size=correct_vocab_size,  # 200 from main vocab.json
        input_dim=model_config['input_dim'],
        d_model=model_config['d_model'],
        encoder_layers=model_config['encoder_layers'],
        decoder_layers=model_config['decoder_layers'],
        num_heads=model_config['num_heads'],
        conv_kernel_size=model_config['conv_kernel_size'],
        dropout=model_config['dropout']
    )
    
    print(f"✅ Model created with vocab_size={correct_vocab_size}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Save corrected model
    print(f"\n4. SAVING CORRECTED MODEL:")
    print("-" * 50)
    
    checkpoint_dir = Path('checkpoints/corrected_vocab_path')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config to use correct vocab file
    corrected_config = config.copy()
    corrected_config['data']['vocab_file'] = str(main_vocab_path)
    corrected_config['model']['vocab_size'] = correct_vocab_size
    
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'config': corrected_config,
        'vocab': main_vocab,
        'vocab_file_used': str(main_vocab_path)
    }
    
    checkpoint_path = checkpoint_dir / 'corrected_vocab_model.pt'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Corrected model saved: {checkpoint_path}")
    
    # 5. Create updated YAML config
    print(f"\n5. CREATING UPDATED YAML CONFIG:")
    print("-" * 50)
    
    # Update the YAML config to explicitly point to correct vocab
    corrected_config['data']['vocab_file'] = str(main_vocab_path)
    corrected_config['model']['vocab_size'] = correct_vocab_size
    
    corrected_yaml_path = Path('config/corrected_training_config.yaml')
    with open(corrected_yaml_path, 'w') as f:
        yaml.dump(corrected_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Updated YAML config: {corrected_yaml_path}")
    
    # 6. Provide training instructions
    print(f"\n6. TRAINING INSTRUCTIONS:")
    print("-" * 50)
    
    print("Now you can train with the correct vocabulary:")
    print()
    print("Option A - Local training:")
    print(f"  python training_scripts/train_konkanivani_asr.py \\")
    print(f"    --config {corrected_yaml_path} \\")
    print(f"    --resume {checkpoint_path}")
    print()
    print("Option B - Kaggle training:")
    print(f"  1. Upload {checkpoint_path} to Kaggle")
    print(f"  2. Upload {corrected_yaml_path} to Kaggle")
    print(f"  3. Use the corrected config in your notebook")
    print()
    print("Expected results:")
    print(f"  - Model will have vocab_size={correct_vocab_size}")
    print(f"  - Can predict all {len(main_vocab)-5} characters in your data")
    print(f"  - Should achieve 20-50% accuracy (vs current 1%)")
    
    return checkpoint_path, corrected_yaml_path

if __name__ == '__main__':
    fix_vocab_path_and_retrain()