#!/usr/bin/env python3
"""
Dataset Analysis for KonkaniVani ASR
===================================
Analyze processed dataset and generate reports
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import pandas as pd


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


def create_visualizations(train_analysis, val_analysis, test_analysis, output_dir):
    """Create visualization plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Duration distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Duration histogram
    axes[0, 0].hist([train_analysis['durations'], val_analysis['durations'], test_analysis['durations']], 
                   bins=50, alpha=0.7, label=['Train', 'Val', 'Test'])
    axes[0, 0].set_xlabel('Duration (seconds)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Audio Duration Distribution')
    axes[0, 0].legend()
    
    # Text length distribution
    axes[0, 1].hist([train_analysis['text_lengths'], val_analysis['text_lengths'], test_analysis['text_lengths']], 
                   bins=50, alpha=0.7, label=['Train', 'Val', 'Test'])
    axes[0, 1].set_xlabel('Text Length (characters)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Text Length Distribution')
    axes[0, 1].legend()
    
    # Word count distribution
    axes[1, 0].hist([train_analysis['word_counts'], val_analysis['word_counts'], test_analysis['word_counts']], 
                   bins=30, alpha=0.7, label=['Train', 'Val', 'Test'])
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Word Count Distribution')
    axes[1, 0].legend()
    
    # Category distribution (train only)
    category_counts = Counter(train_analysis['categories'])
    top_categories = dict(category_counts.most_common(10))
    axes[1, 1].bar(range(len(top_categories)), list(top_categories.values()))
    axes[1, 1].set_xticks(range(len(top_categories)))
    axes[1, 1].set_xticklabels(list(top_categories.keys()), rotation=45, ha='right')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Top 10 Categories (Train)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Character frequency
    char_freq = train_analysis['char_counter'].most_common(50)
    chars, freqs = zip(*char_freq)
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(chars)), freqs)
    plt.xticks(range(len(chars)), chars, fontsize=12)
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.title('Top 50 Character Frequencies (Train Set)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'character_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Duration vs Text Length scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(train_analysis['durations'][:5000], train_analysis['text_lengths'][:5000], 
               alpha=0.5, s=1)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Text Length (characters)')
    plt.title('Duration vs Text Length (5K samples)')
    plt.tight_layout()
    plt.savefig(output_dir / 'duration_vs_text_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {output_dir}")


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
    report.append("## Top 20 Characters (Train Set)")
    report.append("")
    char_freq = train_analysis['char_counter'].most_common(20)
    for i, (char, freq) in enumerate(char_freq, 1):
        report.append(f"{i:2d}. '{char}' - {freq:,} occurrences")
    
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
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Analysis report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze processed dataset')
    parser.add_argument('--data_dir', default='data/konkani-combined/manifests',
                       help='Directory containing manifest files')
    parser.add_argument('--output_dir', default='outputs/dataset_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(train_analysis, val_analysis, test_analysis, output_dir)
    
    # Generate report
    print("Generating report...")
    generate_report(train_analysis, val_analysis, test_analysis, output_dir / 'analysis_report.md')
    
    # Save detailed statistics as JSON
    stats = {
        'train': {k: v for k, v in train_analysis.items() if k not in ['char_counter', 'word_counter']},
        'val': {k: v for k, v in val_analysis.items() if k not in ['char_counter', 'word_counter']},
        'test': {k: v for k, v in test_analysis.items() if k not in ['char_counter', 'word_counter']}
    }
    
    with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()