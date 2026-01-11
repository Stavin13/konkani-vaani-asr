#!/usr/bin/env python3
"""
Combine All ASR Datasets for Maximum Training Data
=================================================
Combine raw corpus, existing processed data, and 10k dataset
"""
import json
import random
from pathlib import Path
from collections import Counter
import argparse


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


def normalize_sample(sample, source_name):
    """Normalize sample format across different datasets"""
    normalized = {
        'audio_filepath': sample['audio_filepath'],
        'text': sample['text'],
        'duration': sample['duration'],
        'language': sample.get('language', 'knn_Deva'),
        'source': source_name
    }
    
    # Add optional fields if available
    if 'sample_rate' in sample:
        normalized['sample_rate'] = sample['sample_rate']
    if 'category' in sample:
        normalized['category'] = sample['category']
    if 'speaker' in sample:
        normalized['speaker'] = sample['speaker']
    
    return normalized


def combine_datasets(dataset_configs, output_dir):
    """Combine multiple datasets"""
    
    all_samples = []
    dataset_stats = {}
    
    print("Loading datasets...")
    
    for config in dataset_configs:
        name = config['name']
        base_dir = Path(config['path'])
        
        print(f"\nLoading {name}...")
        
        # Load train/val/test splits
        train_samples = load_manifest(base_dir / 'train.json')
        val_samples = load_manifest(base_dir / 'val.json') 
        test_samples = load_manifest(base_dir / 'test.json')
        
        # Also try manifest files with different names
        if not train_samples:
            train_samples = load_manifest(base_dir / 'train_manifest.json')
        if not val_samples:
            val_samples = load_manifest(base_dir / 'val_manifest.json')
        if not test_samples:
            test_samples = load_manifest(base_dir / 'test_manifest.json')
        
        # Combine all splits from this dataset
        dataset_samples = train_samples + val_samples + test_samples
        
        # Normalize samples
        normalized_samples = [normalize_sample(s, name) for s in dataset_samples]
        
        # Filter valid samples
        valid_samples = []
        for sample in normalized_samples:
            # Basic validation
            if (sample['text'] and len(sample['text'].strip()) > 0 and
                sample['duration'] > 0.5 and sample['duration'] < 30.0):
                valid_samples.append(sample)
        
        all_samples.extend(valid_samples)
        
        # Statistics
        total_duration = sum(s['duration'] for s in valid_samples) / 3600
        dataset_stats[name] = {
            'samples': len(valid_samples),
            'duration_hours': total_duration,
            'avg_duration': sum(s['duration'] for s in valid_samples) / len(valid_samples) if valid_samples else 0
        }
        
        print(f"  {name}: {len(valid_samples):,} samples ({total_duration:.1f}h)")
    
    print(f"\nTotal combined: {len(all_samples):,} samples")
    
    # Shuffle all samples
    random.shuffle(all_samples)
    
    # Create new splits (80/10/10)
    n_train = int(len(all_samples) * 0.8)
    n_val = int(len(all_samples) * 0.1)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    
    # Save combined manifests
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]
    
    print(f"\nSaving combined manifests to {output_path}...")
    
    for split_name, split_samples in splits:
        manifest_path = output_path / f'{split_name}.json'
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        total_hours = sum(s['duration'] for s in split_samples) / 3600
        print(f"  {split_name:5s}: {len(split_samples):6,} samples ({total_hours:5.1f}h)")
    
    # Print source distribution in training set
    print(f"\nSource distribution in training set:")
    source_counts = Counter(s['source'] for s in train_samples)
    for source, count in source_counts.most_common():
        percentage = count / len(train_samples) * 100
        print(f"  {source:20s}: {count:6,} samples ({percentage:5.1f}%)")
    
    # Save metadata
    metadata = {
        'total_samples': len(all_samples),
        'total_duration_hours': sum(s['duration'] for s in all_samples) / 3600,
        'splits': {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples)
        },
        'source_stats': dataset_stats,
        'source_distribution': dict(source_counts)
    }
    
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Combined dataset saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Combine all ASR datasets')
    parser.add_argument('--output_dir', default='data/konkani-mega-dataset/manifests',
                       help='Output directory for combined manifests')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("="*70)
    print("COMBINING ALL KONKANI ASR DATASETS")
    print("="*70)
    
    # Define datasets to combine
    dataset_configs = [
        {
            'name': 'raw-enhanced',
            'path': 'data/konkani-raw-enhanced/manifests'
        },
        {
            'name': 'asr-v0',
            'path': 'data/konkani-asr-v0/splits/manifests'
        },
        {
            'name': '10k-dataset',
            'path': 'data/konkani-10k'
        },
        {
            'name': 'full-dataset',
            'path': 'data/konkani-full'
        }
    ]
    
    # Combine datasets
    output_path = combine_datasets(dataset_configs, args.output_dir)
    
    print("\n" + "="*70)
    print("COMBINATION COMPLETE!")
    print("="*70)
    print(f"\nMega dataset created at: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Create vocabulary: python scripts/create_vocabulary.py --data_dir {args.output_dir}")
    print(f"2. Start training with mega dataset")
    print(f"3. Expected training time: 20-30 hours")
    print(f"4. Expected results: Much better ASR performance!")


if __name__ == '__main__':
    main()