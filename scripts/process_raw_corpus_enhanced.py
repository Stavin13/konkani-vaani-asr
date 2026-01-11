#!/usr/bin/env python3
"""
Enhanced Raw Corpus Processing for KonkaniVani ASR
=================================================
Process all 72K+ audio files with quality checks and optimization
"""
import json
import os
import sys
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_transcript_file(txt_path):
    """Parse transcript file and extract Devanagari text"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract recorded text (Devanagari)
        if 'RECORDED TEXT ::' in content:
            parts = content.split('RECORDED TEXT ::')
            if len(parts) > 1:
                text_part = parts[1].split('TEXT TRANSLITERATION ::')[0]
                text = text_part.strip()
                
                # Basic quality checks
                if len(text) < 2 or len(text) > 200:
                    return None
                
                # Check if text contains mostly valid characters
                valid_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंःँ़ऽ।॥०१२३४५६७८९ ')
                text_chars = set(text)
                if len(text_chars & valid_chars) / len(text_chars) < 0.7:
                    return None
                
                return text
        return None
    except Exception as e:
        logger.debug(f"Error parsing {txt_path}: {e}")
        return None


def process_audio_file(args):
    """Process a single audio file - designed for multiprocessing"""
    wav_path, txt_path, min_duration, max_duration = args
    
    try:
        # Parse transcript
        text = parse_transcript_file(txt_path)
        if not text:
            return None
        
        # Check audio file
        info = sf.info(wav_path)
        duration = info.duration
        
        # Quality filters
        if duration < min_duration or duration > max_duration:
            return None
        
        if info.samplerate < 8000 or info.samplerate > 48000:
            return None
        
        # Load a small sample to check for corruption
        try:
            data, sr = sf.read(wav_path, frames=1024)
            if np.isnan(data).any() or np.isinf(data).any():
                return None
        except:
            return None
        
        return {
            'audio_filepath': str(wav_path.absolute()),
            'text': text,
            'duration': duration,
            'sample_rate': info.samplerate,
            'language': 'knn_Deva',
            'channels': info.channels,
            'category': wav_path.parent.name
        }
    
    except Exception as e:
        logger.debug(f"Error processing {wav_path}: {e}")
        return None


def collect_audio_files(corpus_dir):
    """Collect all audio-transcript pairs"""
    corpus_path = Path(corpus_dir)
    
    logger.info("Scanning for audio files...")
    wav_files = list(corpus_path.rglob('*.wav'))
    logger.info(f"Found {len(wav_files):,} audio files")
    
    # Find matching transcript files
    file_pairs = []
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix('.txt')
        if txt_path.exists():
            file_pairs.append((wav_path, txt_path))
    
    logger.info(f"Found {len(file_pairs):,} audio-transcript pairs")
    return file_pairs


def process_corpus_parallel(corpus_dir, output_dir, min_duration=0.5, max_duration=30.0, 
                          num_workers=None, batch_size=1000):
    """Process corpus using parallel processing"""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Limit to 8 to avoid overwhelming system
    
    logger.info(f"Using {num_workers} workers for processing")
    
    # Collect files
    file_pairs = collect_audio_files(corpus_dir)
    
    # Prepare arguments for parallel processing
    process_args = [(wav_path, txt_path, min_duration, max_duration) 
                   for wav_path, txt_path in file_pairs]
    
    # Process in batches to manage memory
    all_samples = []
    total_batches = (len(process_args) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(process_args):,} files in {total_batches} batches...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(process_args))
        batch_args = process_args[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_args)} files)")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_audio_file, args) for args in batch_args]
            
            # Collect results with progress bar
            batch_samples = []
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc=f"Batch {batch_idx + 1}"):
                result = future.result()
                if result is not None:
                    batch_samples.append(result)
        
        all_samples.extend(batch_samples)
        logger.info(f"Batch {batch_idx + 1} completed: {len(batch_samples)} valid samples")
    
    logger.info(f"Processing complete: {len(all_samples):,} valid samples from {len(file_pairs):,} files")
    
    return all_samples


def create_balanced_splits(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create balanced train/val/test splits"""
    
    # Group by category for balanced splitting
    category_samples = {}
    for sample in samples:
        category = sample.get('category', 'unknown')
        if category not in category_samples:
            category_samples[category] = []
        category_samples[category].append(sample)
    
    logger.info("Category distribution:")
    for category, cat_samples in sorted(category_samples.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {category:40s}: {len(cat_samples):6,} samples")
    
    # Split each category proportionally
    train_samples, val_samples, test_samples = [], [], []
    
    for category, cat_samples in category_samples.items():
        random.shuffle(cat_samples)
        
        n_train = int(len(cat_samples) * train_ratio)
        n_val = int(len(cat_samples) * val_ratio)
        
        train_samples.extend(cat_samples[:n_train])
        val_samples.extend(cat_samples[n_train:n_train + n_val])
        test_samples.extend(cat_samples[n_train + n_val:])
    
    # Final shuffle
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


def save_manifests(train_samples, val_samples, test_samples, output_dir):
    """Save manifest files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]
    
    logger.info("Saving manifests...")
    
    for split_name, split_samples in splits:
        manifest_path = output_path / f'{split_name}.json'
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        total_hours = sum(s['duration'] for s in split_samples) / 3600
        logger.info(f"  {split_name:5s}: {len(split_samples):6,} samples ({total_hours:5.1f}h)")
    
    return output_path


def print_statistics(samples):
    """Print dataset statistics"""
    logger.info("="*70)
    logger.info("DATASET STATISTICS")
    logger.info("="*70)
    
    # Duration stats
    durations = [s['duration'] for s in samples]
    total_hours = sum(durations) / 3600
    
    logger.info(f"\nAudio Statistics:")
    logger.info(f"  Total samples: {len(samples):,}")
    logger.info(f"  Total duration: {total_hours:.1f} hours")
    logger.info(f"  Average duration: {np.mean(durations):.1f}s")
    logger.info(f"  Median duration: {np.median(durations):.1f}s")
    logger.info(f"  Min duration: {min(durations):.1f}s")
    logger.info(f"  Max duration: {max(durations):.1f}s")
    
    # Text stats
    text_lengths = [len(s['text']) for s in samples]
    logger.info(f"\nText Statistics:")
    logger.info(f"  Average length: {np.mean(text_lengths):.0f} characters")
    logger.info(f"  Median length: {np.median(text_lengths):.0f} characters")
    logger.info(f"  Min length: {min(text_lengths)} characters")
    logger.info(f"  Max length: {max(text_lengths)} characters")
    
    # Sample rate distribution
    sample_rates = {}
    for s in samples:
        sr = s['sample_rate']
        sample_rates[sr] = sample_rates.get(sr, 0) + 1
    
    logger.info(f"\nSample Rate Distribution:")
    for sr, count in sorted(sample_rates.items()):
        logger.info(f"  {sr:5d} Hz: {count:6,} samples ({count/len(samples)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Enhanced raw corpus processing')
    parser.add_argument('--corpus_dir', default='KonkaniRawSpeechCorpus/Data',
                       help='Path to raw corpus directory')
    parser.add_argument('--output_dir', default='data/konkani-raw-enhanced',
                       help='Output directory for processed manifests')
    parser.add_argument('--min_duration', type=float, default=0.5,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=30.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("="*70)
    logger.info("ENHANCED RAW CORPUS PROCESSING")
    logger.info("="*70)
    logger.info(f"Corpus directory: {args.corpus_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Duration range: {args.min_duration}s - {args.max_duration}s")
    
    # Process corpus
    samples = process_corpus_parallel(
        args.corpus_dir,
        args.output_dir,
        args.min_duration,
        args.max_duration,
        args.num_workers,
        args.batch_size
    )
    
    if not samples:
        logger.error("No valid samples found!")
        return 1
    
    # Print statistics
    print_statistics(samples)
    
    # Create splits
    logger.info("\nCreating balanced splits...")
    train_samples, val_samples, test_samples = create_balanced_splits(samples)
    
    # Save manifests
    manifest_dir = save_manifests(train_samples, val_samples, test_samples, 
                                 Path(args.output_dir) / 'manifests')
    
    logger.info("="*70)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Manifests saved to: {manifest_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review data quality: python scripts/analyze_dataset.py")
    logger.info(f"2. Create vocabulary: python scripts/create_vocabulary.py")
    logger.info(f"3. Start training with enhanced dataset")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())