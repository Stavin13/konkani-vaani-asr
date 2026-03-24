#!/usr/bin/env python3
"""
Plot Beam Search Comparison Results
====================================
Visualize CER/WER improvements from different decoding strategies.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_comparison(results, output_path='outputs/beam_search_comparison.png'):
    """Create comparison plots"""
    
    # Extract data
    strategies = list(results['summary'].keys())
    strategy_labels = {
        'greedy': 'Greedy\n(Baseline)',
        'beam_no_lm': 'Beam Search\n(No LM)',
        'beam_3gram': 'Beam Search\n+ 3-gram LM',
        'beam_4gram': 'Beam Search\n+ 4-gram LM'
    }
    
    cer_values = [results['summary'][s]['cer'] for s in strategies]
    wer_values = [results['summary'][s]['wer'] for s in strategies]
    time_values = [results['summary'][s]['time'] for s in strategies]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beam Search + Language Model Comparison', fontsize=16, fontweight='bold')
    
    # Colors
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # 1. CER Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(strategies)), cer_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Decoding Strategy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Character Error Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('CER Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels([strategy_labels[s] for s in strategies], fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, cer_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add baseline line
    baseline_cer = cer_values[0]
    ax1.axhline(y=baseline_cer, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax1.legend()
    
    # 2. WER Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(strategies)), wer_values, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Decoding Strategy', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Word Error Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('WER Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([strategy_labels[s] for s in strategies], fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars2, wer_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add baseline line
    baseline_wer = wer_values[0]
    ax2.axhline(y=baseline_wer, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    
    # 3. Relative Improvement
    ax3 = axes[1, 0]
    baseline_cer = cer_values[0]
    improvements = [((baseline_cer - cer) / baseline_cer * 100) for cer in cer_values]
    
    bar_colors = ['gray' if imp <= 0 else 'green' if imp > 0 else 'red' for imp in improvements[1:]]
    bar_colors.insert(0, 'gray')  # Baseline is gray
    
    bars3 = ax3.bar(range(len(strategies)), improvements, color=bar_colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Decoding Strategy', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CER Improvement vs Baseline (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Relative CER Improvement', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels([strategy_labels[s] for s in strategies], fontsize=9)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars3, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va=va, fontsize=10, fontweight='bold')
    
    # 4. Inference Time
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(strategies)), time_values, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Decoding Strategy', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Inference Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Inference Time Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels([strategy_labels[s] for s in strategies], fontsize=9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars4, time_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add speed multiplier
    baseline_time = time_values[0]
    for i, (bar, val) in enumerate(zip(bars4, time_values)):
        if i > 0:
            speed = baseline_time / val if val > 0 else 0
            ax4.text(bar.get_x() + bar.get_width()/2., val/2,
                    f'{speed:.2f}x', ha='center', va='center', 
                    fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close()

def print_summary(results):
    """Print text summary"""
    print("\n" + "="*80)
    print("BEAM SEARCH COMPARISON SUMMARY")
    print("="*80)
    
    strategies = list(results['summary'].keys())
    
    print("\nMetrics:")
    print("-" * 80)
    print(f"{'Strategy':<25} {'CER':<12} {'WER':<12} {'Time (s)':<12}")
    print("-" * 80)
    
    for strategy in strategies:
        data = results['summary'][strategy]
        strategy_name = {
            'greedy': 'Greedy (Baseline)',
            'beam_no_lm': 'Beam (No LM)',
            'beam_3gram': 'Beam + 3-gram',
            'beam_4gram': 'Beam + 4-gram'
        }[strategy]
        
        print(f"{strategy_name:<25} {data['cer']:>6.2f}%     {data['wer']:>6.2f}%     {data['time']:>8.2f}")
    
    print("\n" + "="*80)
    print("IMPROVEMENTS vs BASELINE")
    print("="*80)
    
    baseline_cer = results['summary']['greedy']['cer']
    baseline_wer = results['summary']['greedy']['wer']
    
    for strategy in strategies[1:]:  # Skip baseline
        data = results['summary'][strategy]
        strategy_name = {
            'beam_no_lm': 'Beam (No LM)',
            'beam_3gram': 'Beam + 3-gram',
            'beam_4gram': 'Beam + 4-gram'
        }[strategy]
        
        cer_imp = ((baseline_cer - data['cer']) / baseline_cer) * 100
        wer_imp = ((baseline_wer - data['wer']) / baseline_wer) * 100
        
        print(f"\n{strategy_name}:")
        print(f"  CER: {cer_imp:+.2f}% {'better' if cer_imp > 0 else 'worse'}")
        print(f"  WER: {wer_imp:+.2f}% {'better' if wer_imp > 0 else 'worse'}")
    
    print("\n" + "="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot beam search comparison results')
    parser.add_argument('--input', type=str, default='outputs/beam_search_comparison.json',
                        help='Input JSON file with results')
    parser.add_argument('--output', type=str, default='outputs/beam_search_comparison.png',
                        help='Output plot file')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("\nRun this first:")
        print("  python3 scripts/test_beam_search_improvements.py --max-samples 50")
        return
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    
    # Print summary
    print_summary(results)
    
    # Create plot
    print("\nCreating visualization...")
    plot_comparison(results, args.output)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
