#!/usr/bin/env python3
"""
Prepare the best_model (1).pt checkpoint for Kaggle upload
"""
import torch
import json
import shutil
from pathlib import Path

def prepare_checkpoint_for_kaggle():
    """
    Prepare the checkpoint for Kaggle by:
    1. Copying it to a clean location
    2. Inspecting its contents
    3. Creating a clean version if needed
    """
    
    source_path = "/Volumes/data&proj/konkani/best_model (1).pt"
    
    # Check if source exists
    if not Path(source_path).exists():
        print(f"❌ Source checkpoint not found: {source_path}")
        return
    
    # Create output directory
    output_dir = Path("kaggle_upload_ready")
    output_dir.mkdir(exist_ok=True)
    
    # Copy checkpoint
    dest_path = output_dir / "best_model.pt"  # Remove spaces for Kaggle
    
    print(f"📂 Copying checkpoint...")
    print(f"  From: {source_path}")
    print(f"  To: {dest_path}")
    
    shutil.copy2(source_path, dest_path)
    
    # Inspect the copied checkpoint
    print(f"\n🔍 Inspecting checkpoint...")
    checkpoint = torch.load(dest_path, map_location='cpu')
    
    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Check model state
    model_state = checkpoint['model_state_dict']
    has_module_prefix = any(key.startswith('module.') for key in model_state.keys())
    
    print(f"🔧 Has DataParallel wrapper: {has_module_prefix}")
    
    # Get basic info
    epoch = checkpoint.get('epoch', 'Unknown')
    vocab_size = checkpoint.get('vocab_size', 'Unknown')
    has_vocab = 'char_to_idx' in checkpoint
    
    print(f"📈 Last epoch: {epoch}")
    print(f"📚 Vocab size: {vocab_size}")
    print(f"📚 Has vocabulary: {has_vocab}")
    
    # Create a clean version if needed
    if has_module_prefix:
        print(f"\n🧹 Creating clean version without DataParallel wrapper...")
        
        clean_model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        
        clean_checkpoint = checkpoint.copy()
        clean_checkpoint['model_state_dict'] = clean_model_state
        
        clean_path = output_dir / "best_model_clean.pt"
        torch.save(clean_checkpoint, clean_path)
        
        print(f"✅ Clean checkpoint saved: {clean_path}")
    
    # Create info file
    info = {
        "original_path": str(source_path),
        "epoch": epoch,
        "vocab_size": vocab_size,
        "has_vocabulary": has_vocab,
        "has_dataparallel_wrapper": has_module_prefix,
        "model_keys_sample": list(model_state.keys())[:10],
        "total_model_keys": len(model_state.keys())
    }
    
    info_path = output_dir / "checkpoint_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📄 Checkpoint info saved: {info_path}")
    
    # Create upload instructions
    instructions = f"""
# Kaggle Upload Instructions

## Files to Upload:
1. `best_model.pt` - Main checkpoint file
2. `best_model_clean.pt` - Clean version (if created)
3. `checkpoint_info.json` - Checkpoint information

## Kaggle Dataset Setup:
1. Go to kaggle.com/datasets
2. Click "New Dataset"
3. Upload the files from this directory
4. Name your dataset (e.g., "konkanivani-checkpoint")
5. Make it public or private as needed

## In Your Notebook:
Replace the checkpoint path with:
```python
checkpoint_path = "/kaggle/input/your-dataset-name/best_model.pt"
```

## Checkpoint Info:
- Original path: {source_path}
- Last epoch: {epoch}
- Vocab size: {vocab_size}
- Has vocabulary: {has_vocab}
- Has DataParallel wrapper: {has_module_prefix}

## Next Steps:
1. Upload your model code (konkanivani_asr.py) as a separate dataset
2. Upload your training data as another dataset
3. Use the provided Kaggle notebook template
"""
    
    instructions_path = output_dir / "UPLOAD_INSTRUCTIONS.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"📋 Upload instructions saved: {instructions_path}")
    
    print(f"\n✅ Checkpoint preparation complete!")
    print(f"📁 Files ready for Kaggle upload in: {output_dir}")
    print(f"📋 Read {instructions_path} for upload instructions")

if __name__ == "__main__":
    prepare_checkpoint_for_kaggle()