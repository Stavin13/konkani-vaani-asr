#!/usr/bin/env python3
"""
Fix DataParallel checkpoint by removing 'module.' prefix
"""

import torch
from pathlib import Path

def fix_dataparallel_checkpoint():
    """Remove 'module.' prefix from DataParallel checkpoint"""
    
    checkpoint_path = Path("checkpoints/best_model_scripts1.pt")
    fixed_path = Path("checkpoints/best_model_scripts1_fixed.pt")
    
    print(f"🔧 Fixing DataParallel checkpoint...")
    print(f"📁 Input: {checkpoint_path}")
    print(f"📁 Output: {fixed_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📊 Original checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'Unknown'):.4f}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
    
    # Fix state dict by removing 'module.' prefix
    old_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    
    fixed_count = 0
    for key, value in old_state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
            fixed_count += 1
        else:
            new_state_dict[key] = value
    
    print(f"✅ Fixed {fixed_count} keys by removing 'module.' prefix")
    
    # Update checkpoint
    checkpoint['model_state_dict'] = new_state_dict
    
    # Save fixed checkpoint
    torch.save(checkpoint, fixed_path)
    
    print(f"✅ Fixed checkpoint saved to: {fixed_path}")
    
    # Show sample keys
    sample_keys = list(new_state_dict.keys())[:10]
    print(f"\n📝 Sample fixed keys:")
    for key in sample_keys:
        print(f"  - {key}")
    
    return fixed_path

if __name__ == "__main__":
    fixed_path = fix_dataparallel_checkpoint()
    print(f"\n🎉 Checkpoint fixed successfully!")
    print(f"📁 Use this file: {fixed_path}")