#!/usr/bin/env python3
"""
Standardize All Vocabulary Files
===============================
Update all vocab files to use the uniform vocabulary format
"""
import json
from pathlib import Path
import argparse

# Standard vocabulary format
STANDARD_VOCAB = {
    "char2idx": {
        "<pad>": 0,
        "<blank>": 1,
        "<sos>": 2,
        "<eos>": 3,
        "<unk>": 4,
        " ": 5,
        "!": 6,
        ",": 7,
        "-": 8,
        ".": 9,
        "?": 10,
        "ँ": 11,
        "ं": 12,
        "ः": 13,
        "अ": 14,
        "आ": 15,
        "इ": 16,
        "ई": 17,
        "उ": 18,
        "ऊ": 19,
        "ऋ": 20,
        "ए": 21,
        "ऐ": 22,
        "ऑ": 23,
        "ओ": 24,
        "औ": 25,
        "क": 26,
        "ख": 27,
        "ग": 28,
        "घ": 29,
        "ङ": 30,
        "च": 31,
        "छ": 32,
        "ज": 33,
        "झ": 34,
        "ञ": 35,
        "ट": 36,
        "ठ": 37,
        "ड": 38,
        "ढ": 39,
        "ण": 40,
        "त": 41,
        "थ": 42,
        "द": 43,
        "ध": 44,
        "न": 45,
        "प": 46,
        "फ": 47,
        "ब": 48,
        "भ": 49,
        "म": 50,
        "य": 51,
        "र": 52,
        "ऱ": 53,
        "ल": 54,
        "ळ": 55,
        "व": 56,
        "श": 57,
        "ष": 58,
        "स": 59,
        "ह": 60,
        "ा": 61,
        "ि": 62,
        "ी": 63,
        "ु": 64,
        "ू": 65,
        "ृ": 66,
        "ॅ": 67,
        "े": 68,
        "ै": 69,
        "ॉ": 70,
        "ो": 71,
        "ौ": 72,
        "्": 73,
        "१": 74,
        "२": 75,
        "९": 76,
        "'": 77,
        "'": 78,
        """: 79,
        """: 80
    },
    "idx2char": {
        "0": "<pad>",
        "1": "<blank>",
        "2": "<sos>",
        "3": "<eos>",
        "4": "<unk>",
        "5": " ",
        "6": "!",
        "7": ",",
        "8": "-",
        "9": ".",
        "10": "?",
        "11": "ँ",
        "12": "ं",
        "13": "ः",
        "14": "अ",
        "15": "आ",
        "16": "इ",
        "17": "ई",
        "18": "उ",
        "19": "ऊ",
        "20": "ऋ",
        "21": "ए",
        "22": "ऐ",
        "23": "ऑ",
        "24": "ओ",
        "25": "औ",
        "26": "क",
        "27": "ख",
        "28": "ग",
        "29": "घ",
        "30": "ङ",
        "31": "च",
        "32": "छ",
        "33": "ज",
        "34": "झ",
        "35": "ञ",
        "36": "ट",
        "37": "ठ",
        "38": "ड",
        "39": "ढ",
        "40": "ण",
        "41": "त",
        "42": "थ",
        "43": "द",
        "44": "ध",
        "45": "न",
        "46": "प",
        "47": "फ",
        "48": "ब",
        "49": "भ",
        "50": "म",
        "51": "य",
        "52": "र",
        "53": "ऱ",
        "54": "ल",
        "55": "ळ",
        "56": "व",
        "57": "श",
        "58": "ष",
        "59": "स",
        "60": "ह",
        "61": "ा",
        "62": "ि",
        "63": "ी",
        "64": "ु",
        "65": "ू",
        "66": "ृ",
        "67": "ॅ",
        "68": "े",
        "69": "ै",
        "70": "ॉ",
        "71": "ो",
        "72": "ौ",
        "73": "्",
        "74": "१",
        "75": "२",
        "76": "९",
        "77": "'",
        "78": "'",
        "79": """,
        "80": """
    },
    "vocab_size": 81
}


def create_nemo_vocab_from_standard():
    """Create NeMo-compatible vocabulary from standard format"""
    chars = [''] * STANDARD_VOCAB['vocab_size']
    for char, idx in STANDARD_VOCAB['char2idx'].items():
        chars[idx] = char
    return chars


def find_vocab_files():
    """Find all vocabulary files in the project"""
    vocab_files = []
    
    # Common vocabulary file locations
    search_paths = [
        "data/*/vocab.json",
        "*/vocab.json", 
        "deployment/data/vocab.json",
        "kaggle_retrain_fixed/vocab.json"
    ]
    
    for pattern in search_paths:
        vocab_files.extend(Path('.').glob(pattern))
    
    return list(set(vocab_files))  # Remove duplicates


def update_vocab_file(vocab_path):
    """Update a vocabulary file to standard format"""
    try:
        # Backup original file
        backup_path = vocab_path.with_suffix('.json.backup')
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
        
        # Write standard vocabulary
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(STANDARD_VOCAB, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Updated: {vocab_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to update {vocab_path}: {e}")
        return False


def update_nemo_vocab_file(vocab_path):
    """Update NeMo vocabulary file to standard format"""
    try:
        # Backup original file
        backup_path = vocab_path.with_suffix('.txt.backup')
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
        
        # Write standard NeMo vocabulary
        chars = create_nemo_vocab_from_standard()
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for char in chars:
                f.write(char + '\n')
        
        print(f"✓ Updated NeMo: {vocab_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to update NeMo {vocab_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Standardize all vocabulary files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STANDARDIZING VOCABULARY FILES")
    print("="*60)
    
    # Find all vocab files
    vocab_files = find_vocab_files()
    
    # Also find NeMo vocab files
    nemo_files = []
    for pattern in ["data/*/vocab_nemo.txt", "*/vocab_nemo.txt"]:
        nemo_files.extend(Path('.').glob(pattern))
    
    print(f"\nFound vocabulary files:")
    for vf in vocab_files:
        print(f"  {vf}")
    
    print(f"\nFound NeMo vocabulary files:")
    for nf in nemo_files:
        print(f"  {nf}")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would update {len(vocab_files)} JSON and {len(nemo_files)} NeMo files")
        return
    
    # Update JSON vocabulary files
    print(f"\nUpdating {len(vocab_files)} JSON vocabulary files...")
    success_count = 0
    for vocab_file in vocab_files:
        if update_vocab_file(vocab_file):
            success_count += 1
    
    # Update NeMo vocabulary files
    print(f"\nUpdating {len(nemo_files)} NeMo vocabulary files...")
    nemo_success_count = 0
    for nemo_file in nemo_files:
        if update_nemo_vocab_file(nemo_file):
            nemo_success_count += 1
    
    print(f"\n" + "="*60)
    print("STANDARDIZATION COMPLETE")
    print("="*60)
    print(f"✓ Updated {success_count}/{len(vocab_files)} JSON vocabulary files")
    print(f"✓ Updated {nemo_success_count}/{len(nemo_files)} NeMo vocabulary files")
    
    print(f"\nStandard vocabulary specs:")
    print(f"  - Vocabulary size: {STANDARD_VOCAB['vocab_size']}")
    print(f"  - Special tokens: 5 (<pad>, <blank>, <sos>, <eos>, <unk>)")
    print(f"  - Devanagari characters: 76")
    print(f"  - Format: char2idx and idx2char mappings")
    
    print(f"\nBackup files created with .backup extension")
    print(f"All vocabulary files now use uniform format!")


if __name__ == '__main__':
    main()