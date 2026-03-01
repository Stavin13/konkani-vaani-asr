#!/usr/bin/env python3
"""
Fix manifest file paths for Windows training
Converts Unix/Mac paths to Windows-compatible paths
"""
import json
import os
from pathlib import Path

def find_audio_files():
    """Find all audio files in the project"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = {}
    
    # Search in common directories
    search_dirs = [
        'data/audio',
        'data/audio/synthetic', 
        'KonkaniRawSpeechCorpus',
        'data',
        '.'
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        full_path = os.path.join(root, file)
                        audio_files[file] = full_path.replace('\\', '/')
    
    return audio_files

def fix_manifest_file(manifest_path, output_path=None):
    """Fix paths in a manifest file"""
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")
        return False
    
    if output_path is None:
        output_path = manifest_path.replace('.json', '_fixed.json')
    
    print(f"🔧 Fixing paths in: {manifest_path}")
    
    # Find available audio files
    audio_files = find_audio_files()
    print(f"📁 Found {len(audio_files)} audio files")
    
    fixed_samples = []
    missing_files = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                sample = json.loads(line.strip())
                original_path = sample['audio_filepath']
                
                # Extract filename
                filename = os.path.basename(original_path)
                
                # Try to find the file
                if filename in audio_files:
                    sample['audio_filepath'] = audio_files[filename]
                    sample['original_path'] = original_path  # Keep original for reference
                    fixed_samples.append(sample)
                else:
                    missing_files.append(filename)
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error on line {line_num}: {e}")
                continue
    
    print(f"✅ Fixed {len(fixed_samples)} samples")
    print(f"❌ Missing {len(missing_files)} files")
    
    if missing_files:
        print("Missing files (first 10):")
        for file in missing_files[:10]:
            print(f"  - {file}")
    
    # Write fixed manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in fixed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved fixed manifest: {output_path}")
    return True

def main():
    print("=================================================================")
    print("🔧 FIXING MANIFEST PATHS FOR WINDOWS")
    print("=================================================================\n")
    
    # Fix common manifest files
    manifest_files = [
        'data/konkani-10k/train_manifest.json',
        'data/konkani-10k/val_manifest.json',
        'data/konkani-10k/test_manifest.json',
        'data/konkani-full/train.json',
        'data/konkani-full/val.json',
        'data/nllb_finetuning/train.json',
        'data/nllb_finetuning/val.json',
        'data/konkani-raw-corpus/manifests/train.json',
        'data/konkani-raw-corpus/manifests/val.json'
    ]
    
    fixed_count = 0
    for manifest_file in manifest_files:
        if os.path.exists(manifest_file):
            if fix_manifest_file(manifest_file):
                fixed_count += 1
            print()
    
    print("=================================================================")
    print(f"✅ FIXED {fixed_count} MANIFEST FILES")
    print("=================================================================")
    
    if fixed_count > 0:
        print("Now you can train with fixed paths:")
        print("python finetune_rtx3060.py --train_manifest data/konkani-10k/train_manifest_fixed.json --val_manifest data/konkani-10k/val_manifest_fixed.json")

if __name__ == "__main__":
    main()