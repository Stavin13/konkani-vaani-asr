#!/usr/bin/env python3
"""
Create a zip file with 10K audio files and manifests for Kaggle upload
"""
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


def create_dataset_zip(manifest_dir, output_zip, include_audio=True):
    """
    Create zip file with manifests and audio files
    
    Args:
        manifest_dir: Directory containing manifests
        output_zip: Output zip file path
        include_audio: Whether to include audio files (False for manifests only)
    """
    manifest_dir = Path(manifest_dir)
    output_zip = Path(output_zip)
    
    print("="*70)
    print("CREATING KAGGLE DATASET ZIP")
    print("="*70)
    
    # Collect all audio files from manifests
    audio_files = set()
    manifest_files = list(manifest_dir.glob('*.json'))
    
    print(f"\nReading manifests from: {manifest_dir}")
    
    for manifest_file in manifest_files:
        # Skip vocab and hidden files
        if manifest_file.name == 'vocab.json' or manifest_file.name.startswith('.'):
            continue
        
        print(f"  Reading {manifest_file.name}...")
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                audio_path = Path(entry['audio_filepath'])
                if audio_path.exists():
                    audio_files.add(audio_path)
    
    print(f"\n✓ Found {len(audio_files)} unique audio files")
    
    # Calculate total size
    if include_audio:
        total_size = sum(f.stat().st_size for f in audio_files)
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
    
    # Create zip file
    print(f"\nCreating zip file: {output_zip}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zipf:
        # Add manifests
        print("\nAdding manifests...")
        for manifest_file in manifest_files:
            arcname = f"konkani-10k/{manifest_file.name}"
            zipf.write(manifest_file, arcname)
            print(f"  ✓ {manifest_file.name}")
        
        # Add audio files
        if include_audio:
            print(f"\nAdding {len(audio_files)} audio files...")
            print("(This may take a while...)")
            
            for audio_file in tqdm(sorted(audio_files), desc="Compressing"):
                # Create archive path maintaining directory structure
                # Store relative to KonkaniRawSpeechCorpus
                if 'KonkaniRawSpeechCorpus' in str(audio_file):
                    rel_path = str(audio_file).split('KonkaniRawSpeechCorpus/')[-1]
                    arcname = f"konkani-10k/audio/{rel_path}"
                else:
                    arcname = f"konkani-10k/audio/{audio_file.name}"
                
                zipf.write(audio_file, arcname)
    
    # Get final zip size
    zip_size = output_zip.stat().st_size
    
    print("\n" + "="*70)
    print("✓ ZIP FILE CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput: {output_zip}")
    print(f"Size: {zip_size / (1024**3):.2f} GB")
    
    if include_audio:
        compression_ratio = (1 - zip_size / total_size) * 100
        print(f"Compression: {compression_ratio:.1f}%")
    
    print(f"\nContents:")
    print(f"  - Manifests: {len(manifest_files)} files")
    if include_audio:
        print(f"  - Audio files: {len(audio_files)} files")
    
    print(f"\nNext steps:")
    print(f"  1. Upload {output_zip.name} to Kaggle as a dataset")
    print(f"  2. Create a new Kaggle notebook")
    print(f"  3. Add the dataset to your notebook")
    print(f"  4. Train ASR model for 100 epochs")
    print()


def create_manifest_only_zip(manifest_dir, output_zip):
    """Create a small zip with just manifests (for testing)"""
    manifest_dir = Path(manifest_dir)
    output_zip = Path(output_zip)
    
    print("Creating manifests-only zip (for testing)...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for manifest_file in manifest_dir.glob('*.json'):
            arcname = f"konkani-10k/{manifest_file.name}"
            zipf.write(manifest_file, arcname)
            print(f"  ✓ {manifest_file.name}")
    
    print(f"\n✓ Created: {output_zip}")
    print(f"  Size: {output_zip.stat().st_size / 1024:.1f} KB")


def update_manifest_paths(manifest_dir, audio_base_path='audio'):
    """
    Update manifest files to use relative paths for Kaggle
    
    Args:
        manifest_dir: Directory containing manifests
        audio_base_path: Base path for audio files in zip
    """
    manifest_dir = Path(manifest_dir)
    
    print("\nUpdating manifest paths for Kaggle...")
    
    for manifest_file in manifest_dir.glob('*_manifest.json'):
        print(f"  Processing {manifest_file.name}...")
        
        # Read manifest
        entries = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                
                # Update audio path to relative
                audio_path = Path(entry['audio_filepath'])
                if 'KonkaniRawSpeechCorpus' in str(audio_path):
                    rel_path = str(audio_path).split('KonkaniRawSpeechCorpus/')[-1]
                    entry['audio_filepath'] = f"{audio_base_path}/{rel_path}"
                else:
                    entry['audio_filepath'] = f"{audio_base_path}/{audio_path.name}"
                
                entries.append(entry)
        
        # Write updated manifest
        with open(manifest_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"    ✓ Updated {len(entries)} entries")


def main():
    parser = argparse.ArgumentParser(description='Create Kaggle dataset zip')
    parser.add_argument(
        '--manifest_dir',
        type=str,
        default='data/konkani-10k',
        help='Directory containing manifests'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='konkani_10k_dataset.zip',
        help='Output zip file name'
    )
    parser.add_argument(
        '--manifests_only',
        action='store_true',
        help='Create zip with manifests only (no audio)'
    )
    parser.add_argument(
        '--update_paths',
        action='store_true',
        help='Update manifest paths to relative (for Kaggle)'
    )
    
    args = parser.parse_args()
    
    # Update paths if requested
    if args.update_paths:
        update_manifest_paths(args.manifest_dir)
    
    # Create zip
    if args.manifests_only:
        create_manifest_only_zip(args.manifest_dir, args.output)
    else:
        create_dataset_zip(
            args.manifest_dir,
            args.output,
            include_audio=True
        )


if __name__ == '__main__':
    main()
