#!/usr/bin/env python3
"""
Create upload package for Kaggle fine-tuning
"""
import shutil
import json
from pathlib import Path
import zipfile

def create_kaggle_package():
    """Create a complete package for Kaggle upload"""
    
    print("🚀 Creating Kaggle upload package...")
    
    # Create package directory
    package_dir = Path("kaggle_finetune_package")
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
    
    # 2. Copy mega dataset
    print("📁 Copying mega dataset...")
    mega_dataset_dir = Path("data/konkani-mega-dataset")
    if mega_dataset_dir.exists():
        target_data_dir = package_dir / "data" / "konkani-mega-dataset"
        shutil.copytree(mega_dataset_dir, target_data_dir, dirs_exist_ok=True)
        print(f"   ✅ {mega_dataset_dir} -> {target_data_dir}")
    else:
        print(f"   ❌ {mega_dataset_dir} not found!")
        return
    
    # 3. Create sample audio directory structure
    print("📁 Creating audio directory structure...")
    audio_dir = package_dir / "audio_samples"
    audio_dir.mkdir(exist_ok=True)
    
    # Copy a few sample audio files for testing
    sample_count = 0
    manifest_path = mega_dataset_dir / "manifests" / "train.json"
    
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Only first 10 samples for testing
                    break
                
                try:
                    sample = json.loads(line.strip())
                    audio_path = Path(sample['audio_filepath'])
                    
                    if audio_path.exists():
                        # Create relative path structure
                        rel_path = audio_path.name
                        target_path = audio_dir / rel_path
                        shutil.copy2(audio_path, target_path)
                        sample_count += 1
                except:
                    continue
    
    print(f"   ✅ Copied {sample_count} sample audio files")
    
    # 4. Create README for Kaggle
    readme_content = f"""# KonkaniVani ASR Fine-tuning Dataset

## 📊 Dataset Contents

### Model
- `best_model.pt` - Best checkpoint (val_loss: 2.0637, vocab_size: 81)

### Mega Dataset
- `data/konkani-mega-dataset/manifests/train.json` - 64,106 training samples
- `data/konkani-mega-dataset/manifests/val.json` - 8,013 validation samples  
- `data/konkani-mega-dataset/vocab.json` - 192-character vocabulary

### Audio Samples
- `audio_samples/` - Sample audio files for testing

## 🚀 Usage

1. Upload this dataset to Kaggle
2. Use the notebook: `KAGGLE_FINETUNE_BEST_MODEL_MEGA_DATASET.ipynb`
3. Update paths in notebook to match your dataset

## 📝 Notes

- Original model: vocab_size=81, val_loss=2.0637
- Target: Improve with 80K+ samples, vocab_size=192
- Expected: val_loss < 1.8

## 🔗 Audio Files

**IMPORTANT**: You need to upload the full audio dataset separately.
The manifest files reference audio paths that need to be available.

Audio files are located at:
```
/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/Data/
```

Create a separate Kaggle dataset with these audio files.
"""
    
    readme_path = package_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # 5. Create dataset metadata
    metadata = {
        "title": "KonkaniVani ASR Fine-tuning Package",
        "id": "your-username/konkanivani-asr-finetune",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["audio", "speech-recognition", "konkani", "asr", "fine-tuning"]
    }
    
    with open(package_dir / "dataset-metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 6. Create zip file
    print("📦 Creating zip file...")
    zip_path = Path("kaggle_finetune_package.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
    
    # Calculate sizes
    package_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file())
    zip_size = zip_path.stat().st_size
    
    print(f"\n✅ Package created successfully!")
    print(f"   📁 Directory: {package_dir}")
    print(f"   📦 Zip file: {zip_path}")
    print(f"   💾 Package size: {package_size / 1024 / 1024:.1f} MB")
    print(f"   🗜️  Zip size: {zip_size / 1024 / 1024:.1f} MB")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Upload {zip_path} to Kaggle as a new dataset")
    print(f"   2. Upload the notebook: notebooks/KAGGLE_FINETUNE_BEST_MODEL_MEGA_DATASET.ipynb")
    print(f"   3. Update paths in notebook to match your dataset")
    print(f"   4. Run the notebook!")
    
    return package_dir, zip_path

if __name__ == "__main__":
    create_kaggle_package()