#!/usr/bin/env python3
"""
Extract 10K samples from Konkani Raw Speech Corpus
Creates train/val/test manifests for ASR training
"""
import os
import json
import random
from pathlib import Path
from collections import defaultdict
import wave
import argparse


def parse_transcript_file(txt_path):
    """
    Parse a transcript .txt file to extract metadata and text
    
    Returns:
        dict with 'text', 'transliteration', 'audio_file', etc.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract fields
        data = {}
        
        # Audio filename
        if 'SPEECH FILE NAME ::' in content:
            audio_file = content.split('SPEECH FILE NAME ::')[1].split('\n')[0].strip()
            data['audio_file'] = audio_file
        
        # Recorded text (Devanagari)
        if 'RECORDED TEXT ::' in content:
            text_section = content.split('RECORDED TEXT ::')[1]
            if 'TEXT TRANSLITERATION ::' in text_section:
                text = text_section.split('TEXT TRANSLITERATION ::')[0].strip()
            else:
                text = text_section.strip()
            data['text'] = text
        
        # Transliteration
        if 'TEXT TRANSLITERATION ::' in content:
            trans = content.split('TEXT TRANSLITERATION ::')[1].strip()
            # Remove any trailing content after transliteration
            trans = trans.split('\n\n')[0].strip()
            data['transliteration'] = trans
        
        # Speaker info
        if 'SPEAKER GENDER ::' in content:
            data['gender'] = content.split('SPEAKER GENDER ::')[1].split('\n')[0].strip()
        
        if 'SPEAKER AGE GROUP ::' in content:
            data['age_group'] = content.split('SPEAKER AGE GROUP ::')[1].split('\n')[0].strip()
        
        if 'CONTENT TYPE ::' in content:
            data['content_type'] = content.split('CONTENT TYPE ::')[1].split('\n')[0].strip()
        
        if 'DIALECT ::' in content:
            data['dialect'] = content.split('DIALECT ::')[1].split('\n')[0].strip()
        
        return data
    
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
        return None


def get_audio_duration(wav_path):
    """Get duration of WAV file in seconds"""
    try:
        with wave.open(str(wav_path), 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return None


def collect_corpus_files(corpus_root, categories, max_per_category=None):
    """
    Collect all transcript files from specified categories
    
    Args:
        corpus_root: Path to KonkaniRawSpeechCorpus/Data
        categories: List of category names to include
        max_per_category: Maximum files per category (None = all)
    
    Returns:
        List of (txt_path, wav_path, category) tuples
    """
    corpus_root = Path(corpus_root)
    files = []
    
    for category in categories:
        category_path = corpus_root / category
        if not category_path.exists():
            print(f"Warning: Category not found: {category}")
            continue
        
        # Find all .txt files in this category
        txt_files = list(category_path.rglob('*.txt'))
        
        print(f"Found {len(txt_files)} files in {category}")
        
        # Shuffle for random selection
        random.shuffle(txt_files)
        
        # Limit if specified
        if max_per_category:
            txt_files = txt_files[:max_per_category]
        
        # Match with .wav files
        for txt_path in txt_files:
            wav_path = txt_path.with_suffix('.wav')
            if wav_path.exists():
                files.append((txt_path, wav_path, category))
            else:
                print(f"Warning: No audio for {txt_path.name}")
    
    return files


def create_manifest(files, output_path, corpus_root):
    """
    Create NeMo-style manifest file
    
    Args:
        files: List of (txt_path, wav_path, category) tuples
        output_path: Where to save manifest
        corpus_root: Root path for making relative paths
    """
    corpus_root = Path(corpus_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    manifest_data = []
    skipped = 0
    
    print(f"\nCreating manifest: {output_path}")
    print(f"Processing {len(files)} files...")
    
    for txt_path, wav_path, category in files:
        # Parse transcript
        data = parse_transcript_file(txt_path)
        
        if not data or not data.get('text'):
            skipped += 1
            continue
        
        # Get audio duration
        duration = get_audio_duration(wav_path)
        if duration is None:
            skipped += 1
            continue
        
        # Filter by duration (5-30 seconds is ideal)
        if duration < 1.0 or duration > 60.0:
            skipped += 1
            continue
        
        # Create manifest entry
        entry = {
            'audio_filepath': str(wav_path.absolute()),
            'text': data['text'],
            'duration': duration,
            'category': category,
        }
        
        # Add optional fields
        if 'transliteration' in data:
            entry['transliteration'] = data['transliteration']
        if 'gender' in data:
            entry['gender'] = data['gender']
        if 'age_group' in data:
            entry['age_group'] = data['age_group']
        if 'dialect' in data:
            entry['dialect'] = data['dialect']
        
        manifest_data.append(entry)
    
    # Write manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in manifest_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Created manifest with {len(manifest_data)} entries")
    print(f"  Skipped: {skipped} files")
    
    return manifest_data


def create_vocabulary(manifest_data, output_path):
    """Create vocabulary from manifest data"""
    chars = set()
    
    for entry in manifest_data:
        text = entry['text']
        for char in text:
            chars.add(char)
    
    # Add special tokens
    special_tokens = ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']
    
    # Create char to index mapping
    char_to_idx = {}
    idx = 0
    
    # Add special tokens first
    for token in special_tokens:
        char_to_idx[token] = idx
        idx += 1
    
    # Add regular characters (sorted for consistency)
    for char in sorted(chars):
        if char not in char_to_idx:
            char_to_idx[char] = idx
            idx += 1
    
    # Save vocabulary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'char2idx': char_to_idx}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Created vocabulary with {len(char_to_idx)} characters")
    print(f"  Saved to: {output_path}")
    
    return char_to_idx


def main():
    parser = argparse.ArgumentParser(description='Extract 10K samples from Konkani corpus')
    parser.add_argument(
        '--corpus_root',
        type=str,
        default='KonkaniRawSpeechCorpus/Data',
        help='Path to corpus Data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/konkani-10k',
        help='Output directory for manifests'
    )
    parser.add_argument(
        '--total_samples',
        type=int,
        default=10000,
        help='Total number of samples to extract'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("="*70)
    print("EXTRACTING 10K SAMPLES FROM KONKANI RAW SPEECH CORPUS")
    print("="*70)
    
    # Define categories and their priorities
    categories_config = [
        ('Sentence-S', 6000),  # Full sentences - highest priority
        ('Contemporary Text-T1', 477),  # All contemporary text
        ('Creative Text-T2', 480),  # All creative text
        ('Phonetically Balanced-W4', 2000),  # Phonetically balanced
        ('Most Frequent Word-Part-W3A', 1000),  # Common words
    ]
    
    # Collect files
    all_files = []
    for category, max_count in categories_config:
        files = collect_corpus_files(
            args.corpus_root,
            [category],
            max_per_category=max_count
        )
        all_files.extend(files)
        
        if len(all_files) >= args.total_samples:
            break
    
    # Shuffle and limit to target
    random.shuffle(all_files)
    all_files = all_files[:args.total_samples]
    
    print(f"\n✓ Collected {len(all_files)} files total")
    
    # Split into train/val/test (80/10/10)
    n_train = int(len(all_files) * 0.8)
    n_val = int(len(all_files) * 0.1)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val:   {len(val_files)} samples")
    print(f"  Test:  {len(test_files)} samples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifests
    train_data = create_manifest(
        train_files,
        output_dir / 'train_manifest.json',
        args.corpus_root
    )
    
    val_data = create_manifest(
        val_files,
        output_dir / 'val_manifest.json',
        args.corpus_root
    )
    
    test_data = create_manifest(
        test_files,
        output_dir / 'test_manifest.json',
        args.corpus_root
    )
    
    # Create vocabulary from training data
    vocab = create_vocabulary(
        train_data,
        output_dir / 'vocab.json'
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    # Category distribution
    category_counts = defaultdict(int)
    for entry in train_data + val_data + test_data:
        category_counts[entry['category']] += 1
    
    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category:40s}: {count:5d} samples")
    
    # Duration statistics
    durations = [entry['duration'] for entry in train_data + val_data + test_data]
    print(f"\nDuration statistics:")
    print(f"  Total audio: {sum(durations)/3600:.2f} hours")
    print(f"  Average: {sum(durations)/len(durations):.2f} seconds")
    print(f"  Min: {min(durations):.2f} seconds")
    print(f"  Max: {max(durations):.2f} seconds")
    
    # Text length statistics
    text_lengths = [len(entry['text']) for entry in train_data + val_data + test_data]
    print(f"\nText length statistics:")
    print(f"  Average: {sum(text_lengths)/len(text_lengths):.1f} characters")
    print(f"  Min: {min(text_lengths)} characters")
    print(f"  Max: {max(text_lengths)} characters")
    
    print("\n" + "="*70)
    print("✓ EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review manifests in {output_dir}")
    print(f"  2. Upload to Kaggle dataset")
    print(f"  3. Train ASR model for 100 epochs (~18 hours)")
    print(f"  4. Test model - should see actual Konkani text!")
    print()


if __name__ == '__main__':
    main()
