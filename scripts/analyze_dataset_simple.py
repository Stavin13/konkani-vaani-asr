#!/usr/bin/env python3
"""
Simple Dataset Analysis for KonkaniVani ASR
==========================================
Analyze processed dataset without visualizations
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import argparse


def load_manifest(manifest_path):
    """Load manifest file"""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def analyze_text_distribution(samples):
    """Analyze text characteristics"""
    texts = [s['text'] for s in samples]
    
    # Character frequency
    char_counter = Counter()
    for text in texts:
        char_counter.update(text)
    
    # Word frequency (space-separated)
    word_counter = Counter()
    for text in texts:
        words = text.split()
        word_counter.update(words)
    
    # Text length distribution
    text_lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    
    return {
        'char_counter': char_counter,
        'word_counter': word_counter,
        'text_lengths': text_lengths,
        'word_counts': word_counts,
        'unique_chars': len(char_counter),
        'unique_words': len(word_counter),
        'total_chars': sum(char_counter.values()),
        'total_words': sum(word_counter.values())
    }


def analyze_audio_distribution(samples):
    """Analyze audio characteristics"""
    durations = [s['duration'] for s in samples]
    sample_rates = [s['sample_rate'] for s in samples]
    categories = [s.get('category', 'unknown') for s in samples]
    
    return {
        'durations': durations,
        'sample_rates': sample_rates,
        'categories': categories,
        'total_hours': sum(durations) / 3600,
        'avg_duration': np.mean(durations),
        'median_duration': np.median(durations)
    }


def generate_report(train_analysis, val_analysis, test_analysis, output_path):
    """Generate detailed analysis report"""
    
    report = []
    report.append("# KonkaniVani ASR Dataset Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Dataset overview
    report.append("## Dataset Overview")
    report.append("")
    total_samples = len(train_analysis['durations']) + len(val_analysis['durations']) + len(test_analysis['durations'])
    total_hours = (train_analysis['total_hours'] + val_analysis['total_hours'] + test_analysis['total_hours'])
    
    report.append(f"- **Total Samples**: {total_samples:,}")
    report.append(f"- **Total Duration**: {total_hours:.1f} hours")
    report.append(f"- **Train**: {len(train_analysis['durations']):,} samples ({train_analysis['total_hours']:.1f}h)")
    report.append(f"- **Validation**: {len(val_analysis['durations']):,} samples ({val_analysis['total_hours']:.1f}h)")
    report.append(f"- **Test**: {len(test_analysis['durations']):,} samples ({test_analysis['total_hours']:.1f}h)")
    report.append("")
    
    # Audio statistics
    report.append("## Audio Statistics")
    report.append("")
    report.append("| Split | Avg Duration | Median Duration | Min | Max |")
    report.append("|-------|--------------|-----------------|-----|-----|")
    
    for split_name, analysis in [('Train', train_analysis), ('Val', val_analysis), ('Test', test_analysis)]:
        durations = analysis['durations']
        report.append(f"| {split_name} | {np.mean(durations):.1f}s | {np.median(durations):.1f}s | {min(durations):.1f}s | {max(durations):.1f}s |")
    
    report.append("")
    
    # Text statistics
    report.append("## Text Statistics")
    report.append("")
    report.append("| Split | Unique Chars | Unique Words | Avg Text Length | Avg Word Count |")
    report.append("|-------|--------------|--------------|-----------------|----------------|")
    
    for split_name, analysis in [('Train', train_analysis), ('Val', val_analysis), ('Test', test_analysis)]:
        report.append(f"| {split_name} | {analysis['unique_chars']} | {analysis['unique_words']} | {np.mean(analysis['text_lengths']):.1f} | {np.mean(analysis['word_counts']):.1f} |")
    
    report.append("")
    
    # Character frequency
    report.append("## Top 30 Characters (Train Set)")
    report.append("")
    char_freq = train_analysis['char_counter'].most_common(30)
    for i, (char, freq) in enumerate(char_freq, 1):
        char_display = repr(char) if char in [' ', '\n', '\t'] else char
        report.append(f"{i:2d}. {char_display:10s} - {freq:,} occurrences")
    
    report.append("")
    
    # Category distribution
    report.append("## Category Distribution (Train Set)")
    report.append("")
    category_counts = Counter(train_analysis['categories'])
    for category, count in category_counts.most_common():
        percentage = count / len(train_analysis['durations']) * 100
        report.append(f"- **{category}**: {count:,} samples ({percentage:.1f}%)")
    
    report.append("")
    
    # Sample rate distribution
    report.append("## Sample Rate Distribution")
    report.append("")
    sr_counts = Counter(train_analysis['sample_rates'])
    for sr, count in sorted(sr_counts.items()):
        percentage = count / len(train_analysis['sample_rates']) * 100
        report.append(f"- **{sr} Hz**: {count:,} samples ({percentage:.1f}%)")
    
    report.append("")
    
    # Duration distribution
    report.append("## Duration Distribution (Train Set)")
    report.append("")
    durations = train_analysis['durations']
    duration_ranges = [
        (0, 2, "0-2s"),
        (2, 5, "2-5s"),
        (5, 10, "5-10s"),
        (10, 20, "10-20s"),
        (20, float('inf'), "20s+")
    ]
    
    for min_dur, max_dur, label in duration_ranges:
        count = sum(1 for d in durations if min_dur <= d < max_dur)
        percentage = count / len(durations) * 100
        report.append(f"- **{label}**: {count:,} samples ({percentage:.1f}%)")
    
    report.append("")
    
    # Text length distribution
    report.append("## Text Length Distribution (Train Set)")
    report.append("")
    text_lengths = train_analysis['text_lengths']
    length_ranges = [
        (0, 5, "0-5 chars"),
        (5, 10, "5-10 chars"),
        (10, 20, "10-20 chars"),
        (20, 50, "20-50 chars"),
        (50, float('inf'), "50+ chars")
    ]
    
    for min_len, max_len, label in length_ranges:
        count = sum(1 for l in text_lengths if min_len <= l < max_len)
        percentage = count / len(text_lengths) * 100
        report.append(f"- **{label}**: {count:,} samples ({percentage:.1f}%)")
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze processed dataset')
    parser.add_argument('--data_dir', default='data/konkani-raw-enhanced/manifests',
                       help='Directory containing manifest files')
    parser.add_argument('--output_dir', default='outputs/dataset_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    print("Loading datasets...")
    
    # Load manifests
    train_samples = load_manifest(data_dir / 'train.json')
    val_samples = load_manifest(data_dir / 'val.json')
    test_samples = load_manifest(data_dir / 'test.json')
    
    print(f"Loaded {len(train_samples):,} train, {len(val_samples):,} val, {len(test_samples):,} test samples")
    
    # Analyze each split
    print("Analyzing datasets...")
    train_analysis = {
        **analyze_text_distribution(train_samples),
        **analyze_audio_distribution(train_samples)
    }
    
    val_analysis = {
        **analyze_text_distribution(val_samples),
        **analyze_audio_distribution(val_samples)
    }
    
    test_analysis = {
        **analyze_text_distribution(test_samples),
        **analyze_audio_distribution(test_samples)
    }
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_samples = len(train_samples) + len(val_samples) + len(test_samples)
    total_hours = train_analysis['total_hours'] + val_analysis['total_hours'] + test_analysis['total_hours']
    
    print(f"\nDataset Overview:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total duration: {total_hours:.1f} hours")
    print(f"  Train: {len(train_samples):,} samples ({train_analysis['total_hours']:.1f}h)")
    print(f"  Val: {len(val_samples):,} samples ({val_analysis['total_hours']:.1f}h)")
    print(f"  Test: {len(test_samples):,} samples ({test_analysis['total_hours']:.1f}h)")
    
    print(f"\nText Statistics (Train):")
    print(f"  Unique characters: {train_analysis['unique_chars']}")
    print(f"  Unique words: {train_analysis['unique_words']:,}")
    print(f"  Avg text length: {np.mean(train_analysis['text_lengths']):.1f} chars")
    print(f"  Avg word count: {np.mean(train_analysis['word_counts']):.1f} words")
    
    print(f"\nAudio Statistics (Train):")
    print(f"  Avg duration: {train_analysis['avg_duration']:.1f}s")
    print(f"  Median duration: {train_analysis['median_duration']:.1f}s")
    print(f"  Duration range: {min(train_analysis['durations']):.1f}s - {max(train_analysis['durations']):.1f}s")
    
    # Top categories
    print(f"\nTop Categories (Train):")
    category_counts = Counter(train_analysis['categories'])
    for category, count in category_counts.most_common(5):
        percentage = count / len(train_samples) * 100
        print(f"  {category:30s}: {count:6,} samples ({percentage:5.1f}%)")
    
    # Generate detailed report
    print("\nGenerating detailed report...")
    generate_report(train_analysis, val_analysis, test_analysis, output_dir / 'analysis_report.md')
    
    # Save detailed statistics as JSON
    stats = {
        'train': {k: v for k, v in train_analysis.items() if k not in ['char_counter', 'word_counter']},
        'val': {k: v for k, v in val_analysis.items() if k not in ['char_counter', 'word_counter']},
        'test': {k: v for k, v in test_analysis.items() if k not in ['char_counter', 'word_counter']}
    }
    
    with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✓ Analysis complete! Results saved to {output_dir}")
    print(f"\nNext step: Create vocabulary with:")
    print(f"  python scripts/create_vocabulary.py --data_dir {args.data_dir}")


if __name__ == '__main__':
    main()