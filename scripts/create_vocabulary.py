#!/usr/bin/env python3
"""
Create Vocabulary for KonkaniVani ASR
====================================
Generate character-level vocabulary from processed dataset
"""
import json
import argparse
from pathlib import Path
from collections import Counter
import unicodedata


def load_manifest(manifest_path):
    """Load manifest file"""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_characters(samples):
    """Extract all unique characters from text"""
    char_counter = Counter()
    
    for sample in samples:
        text = sample['text']
        # Normalize unicode (NFC normalization)
        text = unicodedata.normalize('NFC', text)
        char_counter.update(text)
    
    return char_counter


def create_vocabulary(char_counter, min_freq=1, include_special_tokens=True):
    """Create vocabulary mapping"""
    
    # Special tokens
    special_tokens = []
    if include_special_tokens:
        special_tokens = [
            '<pad>',    # Padding token
            '<blank>',  # CTC blank token
            '<unk>',    # Unknown token
            '<sos>',    # Start of sequence
            '<eos>',    # End of sequence
        ]
    
    # Filter characters by frequency
    filtered_chars = [char for char, freq in char_counter.items() if freq >= min_freq]
    
    # Sort characters for consistency
    filtered_chars = sorted(filtered_chars)
    
    # Create character to index mapping
    char_to_idx = {}
    idx_to_char = {}
    
    # Add special tokens first
    for i, token in enumerate(special_tokens):
        char_to_idx[token] = i
        idx_to_char[i] = token
    
    # Add regular characters
    for char in filtered_chars:
        if char not in char_to_idx:  # Avoid duplicates
            idx = len(char_to_idx)
            char_to_idx[char] = idx
            idx_to_char[idx] = char
    
    return char_to_idx, idx_to_char, char_counter


def analyze_vocabulary(char_to_idx, char_counter):
    """Analyze vocabulary characteristics"""
    
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    
    print(f"\nVocabulary size: {len(char_to_idx)}")
    
    # Character categories
    categories = {
        'Devanagari Letters': 0,
        'Devanagari Digits': 0,
        'Devanagari Marks': 0,
        'Punctuation': 0,
        'Whitespace': 0,
        'Latin': 0,
        'Other': 0
    }
    
    for char in char_to_idx:
        if char.startswith('<') and char.endswith('>'):
            continue  # Skip special tokens
        
        cat = unicodedata.category(char)
        name = unicodedata.name(char, 'UNKNOWN')
        
        if 'DEVANAGARI' in name and 'LETTER' in name:
            categories['Devanagari Letters'] += 1
        elif 'DEVANAGARI' in name and 'DIGIT' in name:
            categories['Devanagari Digits'] += 1
        elif 'DEVANAGARI' in name:
            categories['Devanagari Marks'] += 1
        elif cat.startswith('P'):  # Punctuation
            categories['Punctuation'] += 1
        elif cat.startswith('Z'):  # Whitespace
            categories['Whitespace'] += 1
        elif ord(char) < 128:  # ASCII/Latin
            categories['Latin'] += 1
        else:
            categories['Other'] += 1
    
    print("\nCharacter categories:")
    for category, count in categories.items():
        if count > 0:
            print(f"  {category:20s}: {count:3d}")
    
    # Most frequent characters
    print("\nTop 20 most frequent characters:")
    for i, (char, freq) in enumerate(char_counter.most_common(20), 1):
        char_display = repr(char) if char in [' ', '\n', '\t'] else char
        print(f"  {i:2d}. {char_display:10s} - {freq:6,} occurrences")
    
    # Character frequency distribution
    freqs = list(char_counter.values())
    print(f"\nFrequency statistics:")
    print(f"  Total characters: {sum(freqs):,}")
    print(f"  Average frequency: {sum(freqs)/len(freqs):.1f}")
    print(f"  Min frequency: {min(freqs)}")
    print(f"  Max frequency: {max(freqs):,}")
    
    # Rare characters (frequency = 1)
    rare_chars = [char for char, freq in char_counter.items() if freq == 1]
    if rare_chars:
        print(f"\nRare characters (frequency = 1): {len(rare_chars)}")
        print("  Examples:", ' '.join(rare_chars[:10]))


def save_vocabulary(char_to_idx, idx_to_char, output_path):
    """Save vocabulary to JSON file"""
    
    vocab_data = {
        'char_to_idx': char_to_idx,
        'idx_to_char': {str(k): v for k, v in idx_to_char.items()},  # JSON keys must be strings
        'vocab_size': len(char_to_idx),
        'special_tokens': ['<pad>', '<blank>', '<unk>', '<sos>', '<eos>']
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Vocabulary saved to: {output_path}")
    return vocab_data


def create_nemo_vocab(char_to_idx, output_path):
    """Create NeMo-compatible vocabulary file"""
    
    # NeMo expects a simple list of characters
    chars = [''] * len(char_to_idx)
    for char, idx in char_to_idx.items():
        chars[idx] = char
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in chars:
            f.write(char + '\n')
    
    print(f"✓ NeMo vocabulary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create vocabulary from dataset')
    parser.add_argument('--data_dir', default='data/konkani-combined/manifests',
                       help='Directory containing manifest files')
    parser.add_argument('--output_dir', default='data/konkani-combined',
                       help='Output directory for vocabulary files')
    parser.add_argument('--min_freq', type=int, default=1,
                       help='Minimum character frequency to include')
    parser.add_argument('--train_only', action='store_true',
                       help='Use only training data for vocabulary')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("CREATING VOCABULARY FOR KONKANIVANI ASR")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_samples = load_manifest(data_dir / 'train.json')
    print(f"Loaded {len(train_samples):,} training samples")
    
    samples = train_samples
    if not args.train_only:
        val_samples = load_manifest(data_dir / 'val.json')
        test_samples = load_manifest(data_dir / 'test.json')
        samples.extend(val_samples)
        samples.extend(test_samples)
        print(f"Loaded {len(val_samples):,} validation and {len(test_samples):,} test samples")
    
    print(f"Total samples for vocabulary: {len(samples):,}")
    
    # Extract characters
    print("\nExtracting characters...")
    char_counter = extract_characters(samples)
    print(f"Found {len(char_counter)} unique characters")
    
    # Create vocabulary
    print(f"\nCreating vocabulary (min_freq={args.min_freq})...")
    char_to_idx, idx_to_char, char_counter = create_vocabulary(
        char_counter, 
        min_freq=args.min_freq
    )
    
    # Analyze vocabulary
    analyze_vocabulary(char_to_idx, char_counter)
    
    # Save vocabulary files
    print("\nSaving vocabulary files...")
    
    # Main vocabulary file
    vocab_data = save_vocabulary(
        char_to_idx, 
        idx_to_char, 
        output_dir / 'vocab.json'
    )
    
    # NeMo-compatible vocabulary
    create_nemo_vocab(
        char_to_idx,
        output_dir / 'vocab_nemo.txt'
    )
    
    # Character frequency file (for analysis)
    freq_data = dict(char_counter.most_common())
    with open(output_dir / 'char_frequencies.json', 'w', encoding='utf-8') as f:
        json.dump(freq_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Character frequencies saved to: {output_dir / 'char_frequencies.json'}")
    
    print("\n" + "="*60)
    print("VOCABULARY CREATION COMPLETE!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {output_dir / 'vocab.json'} (main vocabulary)")
    print(f"  - {output_dir / 'vocab_nemo.txt'} (NeMo format)")
    print(f"  - {output_dir / 'char_frequencies.json'} (character frequencies)")
    
    print(f"\nVocabulary summary:")
    print(f"  - Size: {len(char_to_idx)} characters")
    print(f"  - Special tokens: 5")
    print(f"  - Regular characters: {len(char_to_idx) - 5}")
    print(f"  - Ready for ASR training!")


if __name__ == '__main__':
    main()