#!/usr/bin/env python3
"""
Validate Dataset Against Standard Vocabulary
==========================================
Check if all characters in datasets are covered by standard vocabulary
"""
import json
from pathlib import Path
from collections import Counter
import argparse

# Load standard vocabulary
STANDARD_VOCAB = {
    "char2idx": {
        "<pad>": 0, "<blank>": 1, "<sos>": 2, "<eos>": 3, "<unk>": 4,
        " ": 5, "!": 6, ",": 7, "-": 8, ".": 9, "?": 10,
        "ँ": 11, "ं": 12, "ः": 13, "अ": 14, "आ": 15, "इ": 16, "ई": 17,
        "उ": 18, "ऊ": 19, "ऋ": 20, "ए": 21, "ऐ": 22, "ऑ": 23, "ओ": 24,
        "औ": 25, "क": 26, "ख": 27, "ग": 28, "घ": 29, "ङ": 30, "च": 31,
        "छ": 32, "ज": 33, "झ": 34, "ञ": 35, "ट": 36, "ठ": 37, "ड": 38,
        "ढ": 39, "ण": 40, "त": 41, "थ": 42, "द": 43, "ध": 44, "न": 45,
        "प": 46, "फ": 47, "ब": 48, "भ": 49, "म": 50, "य": 51, "र": 52,
        "ऱ": 53, "ल": 54, "ळ": 55, "व": 56, "श": 57, "ष": 58, "स": 59,
        "ह": 60, "ा": 61, "ि": 62, "ी": 63, "ु": 64, "ू": 65, "ृ": 66,
        "ॅ": 67, "े": 68, "ै": 69, "ॉ": 70, "ो": 71, "ौ": 72, "्": 73,
        "१": 74, "२": 75, "९": 76, "'": 77, "'": 78, """: 79, """: 80
    }
}

STANDARD_CHARS = set(STANDARD_VOCAB['char2idx'].keys())


def load_manifest(manifest_path):
    """Load manifest file"""
    samples = []
    if not manifest_path.exists():
        return samples
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except:
                continue
    return samples


def analyze_dataset_chars(manifest_path):
    """Analyze characters in a dataset"""
    samples = load_manifest(manifest_path)
    
    if not samples:
        return None, None, None
    
    # Extract all characters
    all_chars = Counter()
    for sample in samples:
        text = sample.get('text', '')
        all_chars.update(text)
    
    # Find characters not in standard vocab
    dataset_chars = set(all_chars.keys())
    missing_chars = dataset_chars - STANDARD_CHARS
    covered_chars = dataset_chars & STANDARD_CHARS
    
    return all_chars, missing_chars, covered_chars


def validate_dataset(dataset_path):
    """Validate a dataset against standard vocabulary"""
    dataset_path = Path(dataset_path)
    
    print(f"\nValidating dataset: {dataset_path}")
    print("-" * 50)
    
    # Check train/val/test manifests
    manifest_files = ['train.json', 'val.json', 'test.json']
    
    # Also check alternative names
    alt_names = ['train_manifest.json', 'val_manifest.json', 'test_manifest.json']
    
    all_missing = set()
    all_covered = set()
    total_samples = 0
    
    for manifest_name in manifest_files + alt_names:
        manifest_path = dataset_path / manifest_name
        
        if not manifest_path.exists():
            continue
        
        char_counter, missing_chars, covered_chars = analyze_dataset_chars(manifest_path)
        
        if char_counter is None:
            continue
        
        samples_count = sum(1 for _ in load_manifest(manifest_path))
        total_samples += samples_count
        
        print(f"  {manifest_name:20s}: {samples_count:6,} samples")
        
        if missing_chars:
            print(f"    Missing chars ({len(missing_chars)}): {sorted(missing_chars)}")
            all_missing.update(missing_chars)
        else:
            print(f"    ✓ All characters covered")
        
        all_covered.update(covered_chars)
    
    # Summary
    coverage_pct = len(all_covered) / len(all_covered | all_missing) * 100 if (all_covered | all_missing) else 100
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Characters found: {len(all_covered | all_missing)}")
    print(f"  Characters covered: {len(all_covered)} ({coverage_pct:.1f}%)")
    print(f"  Missing characters: {len(all_missing)}")
    
    if all_missing:
        print(f"  Missing: {sorted(all_missing)}")
        return False
    else:
        print(f"  ✓ Dataset fully compatible with standard vocabulary")
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate datasets against standard vocabulary')
    parser.add_argument('--datasets', nargs='+', 
                       default=[
                           'data/konkani-mega-dataset/manifests',
                           'data/konkani-raw-enhanced/manifests',
                           'data/konkani-10k',
                           'data/konkani-asr-v0/splits/manifests'
                       ],
                       help='Dataset directories to validate')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DATASET VOCABULARY VALIDATION")
    print("="*60)
    print(f"Standard vocabulary size: {len(STANDARD_CHARS)} characters")
    
    all_compatible = True
    
    for dataset_dir in args.datasets:
        dataset_path = Path(dataset_dir)
        if dataset_path.exists():
            is_compatible = validate_dataset(dataset_path)
            all_compatible = all_compatible and is_compatible
        else:
            print(f"\n⚠️  Dataset not found: {dataset_path}")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if all_compatible:
        print("✅ All datasets are compatible with standard vocabulary!")
        print("   Ready for training with uniform vocab.")
    else:
        print("❌ Some datasets have characters not in standard vocabulary.")
        print("   Consider updating datasets or expanding vocabulary.")
    
    print(f"\nStandard vocabulary covers:")
    print(f"  - Special tokens: 5")
    print(f"  - Devanagari letters: 47")
    print(f"  - Devanagari marks: 13") 
    print(f"  - Digits: 3")
    print(f"  - Punctuation: 13")
    print(f"  - Total: 81 characters")


if __name__ == '__main__':
    main()