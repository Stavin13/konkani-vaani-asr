#!/usr/bin/env python3
"""
Create a quick Kaggle package for immediate testing (no audio files)
"""
import shutil
import json
from pathlib import Path
import zipfile

def create_quick_package():
    """Create a lightweight package for immediate Kaggle testing"""
    
    print("🚀 Creating quick Kaggle package (metadata only)...")
    
    # Create package directory
    package_dir = Path("kaggle_quick_package")
    package_dir.mkdir(exist_ok=True)
    
    # 1. Copy best model
    print("📁 Copying best model...")
    best_model = Path("best_model (1).pt")
    if best_model.exists():
        shutil.copy2(best_model, package_dir / "best_model.pt")
        print(f"   ✅ {best_model} -> {package_dir}/best_model.pt")
    else:
        print(f"   ❌ {best_model} not found!")
        return
    
    # 2. Copy mega dataset metadata
    print("📁 Copying mega dataset metadata...")
    mega_dataset_dir = Path("data/konkani-mega-dataset")
    if mega_dataset_dir.exists():
        target_data_dir = package_dir / "data" / "konkani-mega-dataset"
        
        # Only copy manifests and vocab (no audio)
        target_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy manifests
        manifests_dir = mega_dataset_dir / "manifests"
        if manifests_dir.exists():
            target_manifests = target_data_dir / "manifests"
            shutil.copytree(manifests_dir, target_manifests, dirs_exist_ok=True)
            print(f"   ✅ Manifests copied")
        
        # Copy vocabulary files
        for vocab_file in ["vocab.json", "vocab_nemo.txt", "char_frequencies.json"]:
            src_file = mega_dataset_dir / vocab_file
            if src_file.exists():
                shutil.copy2(src_file, target_data_dir / vocab_file)
                print(f"   ✅ {vocab_file} copied")
    else:
        print(f"   ❌ {mega_dataset_dir} not found!")
        return
    
    # 3. Create README
    readme_content = f"""# KonkaniVani ASR Quick Test Package

## 📊 Contents

### Model
- `best_model.pt` - Best checkpoint (val_loss: 2.0637, vocab_size: 81)

### Dataset Metadata
- `data/konkani-mega-dataset/manifests/train.json` - 64,106 training samples
- `data/konkani-mega-dataset/manifests/val.json` - 8,013 validation samples  
- `data/konkani-mega-dataset/vocab.json` - 81-character vocabulary

## ⚠️ Audio Files Not Included

This package contains only metadata for quick testing. 
Audio files need to be uploaded separately or paths updated in the notebook.

## 🚀 Quick Start

1. Upload this dataset to Kaggle
2. Use notebook: `KAGGLE_FINETUNE_BEST_MODEL_MEGA_DATASET.ipynb`
3. Update audio paths or use subset for testing

## 📝 Next Steps

For full training:
1. Upload audio files separately
2. Update manifest paths in notebook
3. Or start with subset testing
"""
    
    readme_path = package_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # 4. Create dataset metadata
    metadata = {
        "title": "KonkaniVani ASR Quick Test Package",
        "id": "your-username/konkanivani-asr-quick-test",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["audio", "speech-recognition", "konkani", "asr", "fine-tuning", "test"]
    }
    
    with open(package_dir / "dataset-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 5. Create zip file
    print("📦 Creating zip file...")
    zip_path = Path("kaggle_quick_package.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
    
    # Calculate sizes
    package_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file())
    zip_size = zip_path.stat().st_size
    
    print(f"\n✅ Quick package created!")
    print(f"   📁 Directory: {package_dir}")
    print(f"   📦 Zip file: {zip_path}")
    print(f"   💾 Package size: {package_size / 1024 / 1024:.1f} MB")
    print(f"   🗜️  Zip size: {zip_size / 1024 / 1024:.1f} MB")
    print(f"   ⏱️  Creation time: ~30 seconds")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Upload {zip_path} to Kaggle (should take 1-2 minutes)")
    print(f"   2. Upload notebook and test with subset")
    print(f"   3. Upload audio files later for full training")
    
    return package_dir, zip_path

if __name__ == "__main__":
    create_quick_package()