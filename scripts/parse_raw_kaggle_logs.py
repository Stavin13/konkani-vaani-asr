"""
Parse Raw Kaggle Training Logs and Generate Comprehensive Graphs
Handles the raw tqdm progress bar output format from Kaggle notebooks
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_raw_kaggle_logs(log_file):
    """Parse raw Kaggle training logs with tqdm progress bars"""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern for epoch progress lines: Epoch X: XX%|... | step/total [..., loss=X.X, ctc=X.X]
    progress_pattern = r'Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s+\[[^\]]+loss=([\d.]+),\s*ctc=([\d.]+)\]'
    
    # Pattern for epoch summary: Epoch X/Y Train Loss: X.X (CTC: X.X) Val Loss: X.X (CTC: X.X)
    # Also handle format without "Epoch X/Y" prefix
    summary_pattern = r'(?:Epoch (\d+)/\d+\s+)?Train Loss:\s*([\d.]+)\s*\(CTC:\s*([\d.]+)\)\s+Val Loss:\s*([\d.]+)\s*\(CTC:\s*([\d.]+)\)'
    
    # Pattern for validation progress
    val_pattern = r'Validation:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)'
    
    # Collect data
    epoch_data = defaultdict(lambda: {'train_losses': [], 'train_ctc': [], 'steps': []})
    epoch_summaries = []
    
    # Parse progress lines
    for match in re.finditer(progress_pattern, content):
        epoch = int(match.group(1))
        step = int(match.group(2))
        total_steps = int(match.group(3))
        loss = float(match.group(4))
        ctc = float(match.group(5))
        
        epoch_data[epoch]['train_losses'].append(loss)
        epoch_data[epoch]['train_ctc'].append(ctc)
        epoch_data[epoch]['steps'].append(step)
        epoch_data[epoch]['total_steps'] = total_steps
    
    # Parse epoch summaries
    epoch_counter = 1
    for match in re.finditer(summary_pattern, content):
        epoch_str = match.group(1)
        epoch = int(epoch_str) if epoch_str else epoch_counter
        train_loss = float(match.group(2))
        train_ctc = float(match.group(3))
        val_loss = float(match.group(4))
        val_ctc = float(match.group(5))
        
        epoch_summaries.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ctc': train_ctc,
            'val_loss': val_loss,
            'val_ctc': val_ctc
        })
        epoch_counter += 1
    
    if not epoch_summaries and not epoch_data:
        print("❌ No training data found in logs")
        return None
    
    print(f"✅ Found training data:")
    print(f"   Epochs with progress data: {len(epoch_data)}")
    print(f"   Epoch summaries: {len(epoch_summaries)}")
    
    return {
        'epoch_data': dict(epoch_data),
        'summaries': epoch_summaries
    }


def plot_comprehensive_training_curves(data, output_dir='outputs'):
    """Generate comprehensive training visualization"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    epoch_data = data['epoch_data']
    summaries = data['summaries']
    
    if not summaries:
        print("⚠️  No epoch summaries found, plotting progress data only")
        plot_progress_only(epoch_data, output_dir)
        return
    
    # Extract summary data
    epochs = [s['epoch'] for s in summaries]
    train_losses = [s['train_loss'] for s in summaries]
    train_ctc = [s['train_ctc'] for s in summaries]
    val_losses = [s['val_loss'] for s in summaries]
    val_ctc = [s['val_ctc'] for s in summaries]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('KonkaniVani ASR Training Analysis (Kaggle)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Plot 1: Train vs Val Loss (Main)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2.5, 
            color='#2E86AB', marker='o', markersize=6, alpha=0.8)
    ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2.5, 
            color='#A23B72', marker='s', markersize=6, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight best epoch
    best_idx = np.argmin(val_losses)
    best_epoch = epochs[best_idx]
    best_loss = val_losses[best_idx]
    ax1.plot(best_epoch, best_loss, 'r*', markersize=20, 
             label=f'Best: Epoch {best_epoch} ({best_loss:.4f})', zorder=5)
    ax1.legend(fontsize=11, loc='upper right')
    
    # Plot 2: Loss Improvement %
    ax2 = fig.add_subplot(gs[0, 2])
    train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 
                         for loss in train_losses]
    val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 
                       for loss in val_losses]
    
    ax2.plot(epochs, train_improvement, label='Train', linewidth=2, 
            color='#2E86AB', marker='o', markersize=5)
    ax2.plot(epochs, val_improvement, label='Val', linewidth=2, 
            color='#A23B72', marker='s', markersize=5)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Loss Improvement', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 3: CTC Loss Comparison
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(epochs, train_ctc, label='Train CTC', linewidth=2.5, 
            color='#2E86AB', marker='o', markersize=6, alpha=0.8)
    ax3.plot(epochs, val_ctc, label='Val CTC', linewidth=2.5, 
            color='#A23B72', marker='s', markersize=6, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CTC Loss', fontsize=12, fontweight='bold')
    ax3.set_title('CTC Loss (Audio-Text Alignment)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Overfitting Analysis
    ax4 = fig.add_subplot(gs[1, 2])
    gap = [val - train for val, train in zip(val_losses, train_losses)]
    colors = ['green' if g < 1.0 else 'orange' if g < 2.0 else 'red' for g in gap]
    ax4.bar(epochs, gap, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Gap (Val - Train)', fontsize=11, fontweight='bold')
    ax4.set_title('Overfitting Check', fontsize=13, fontweight='bold')
    ax4.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Threshold')
    ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Plot 5: Within-Epoch Progress (if available)
    ax5 = fig.add_subplot(gs[2, :])
    if epoch_data:
        # Plot progress for first and last epoch
        epochs_to_plot = [min(epoch_data.keys()), max(epoch_data.keys())]
        colors_prog = ['#2E86AB', '#A23B72']
        
        for epoch_num, color in zip(epochs_to_plot, colors_prog):
            if epoch_num in epoch_data:
                ed = epoch_data[epoch_num]
                # Sample every Nth point to avoid overcrowding
                sample_rate = max(1, len(ed['steps']) // 50)
                steps = ed['steps'][::sample_rate]
                losses = ed['train_losses'][::sample_rate]
                ax5.plot(steps, losses, label=f'Epoch {epoch_num}', 
                        linewidth=2, color=color, alpha=0.7)
        
        ax5.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax5.set_title('Within-Epoch Training Progress', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Save figure
    output_path = Path(output_dir) / 'kaggle_training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive training analysis to: {output_path}")
    
    # Save PDF version
    pdf_path = Path(output_dir) / 'kaggle_training_analysis.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved PDF version to: {pdf_path}")
    
    plt.close()
    
    # Create additional detailed plots
    plot_detailed_metrics(summaries, epoch_data, output_dir)
    
    # Print summary
    print_detailed_summary(summaries, epoch_data)


def plot_detailed_metrics(summaries, epoch_data, output_dir):
    """Create additional detailed metric plots"""
    
    epochs = [s['epoch'] for s in summaries]
    val_losses = [s['val_loss'] for s in summaries]
    
    # Create figure for detailed validation loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Detailed Validation Loss Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Val Loss with moving average
    ax1.plot(epochs, val_losses, 'o-', linewidth=2, color='#A23B72', 
             label='Val Loss', markersize=6, alpha=0.7)
    
    # Add moving average if enough data
    if len(val_losses) >= 3:
        window = min(3, len(val_losses))
        moving_avg = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        ax1.plot(ma_epochs, moving_avg, '--', linewidth=2.5, color='#F18F01', 
                label=f'{window}-Epoch Moving Avg', alpha=0.9)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Loss Trend', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Loss reduction per epoch
    ax2.bar(epochs[1:], [-np.diff(val_losses)[i] for i in range(len(np.diff(val_losses)))],
            color=['green' if d > 0 else 'red' for d in -np.diff(val_losses)],
            alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss Reduction', fontsize=12, fontweight='bold')
    ax2.set_title('Epoch-to-Epoch Improvement', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'validation_loss_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved detailed validation analysis to: {output_path}")
    plt.close()


def plot_progress_only(epoch_data, output_dir):
    """Plot only progress data when summaries are not available"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Training Progress (No Epoch Summaries Available)', 
                 fontsize=16, fontweight='bold')
    
    for epoch_num in sorted(epoch_data.keys()):
        ed = epoch_data[epoch_num]
        color = plt.cm.viridis(epoch_num / max(epoch_data.keys()))
        
        axes[0].plot(ed['steps'], ed['train_losses'], label=f'Epoch {epoch_num}',
                    linewidth=1.5, color=color, alpha=0.7)
        axes[1].plot(ed['steps'], ed['train_ctc'], label=f'Epoch {epoch_num}',
                    linewidth=1.5, color=color, alpha=0.7)
    
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Progress', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('CTC Loss', fontsize=12)
    axes[1].set_title('CTC Loss Progress', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'training_progress.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved training progress to: {output_path}")
    plt.close()


def print_detailed_summary(summaries, epoch_data):
    """Print comprehensive training summary"""
    print("\n" + "="*80)
    print("KAGGLE TRAINING ANALYSIS SUMMARY")
    print("="*80)
    
    if summaries:
        epochs = [s['epoch'] for s in summaries]
        train_losses = [s['train_loss'] for s in summaries]
        val_losses = [s['val_loss'] for s in summaries]
        train_ctc = [s['train_ctc'] for s in summaries]
        val_ctc = [s['val_ctc'] for s in summaries]
        
        print(f"\n📊 TRAINING OVERVIEW")
        print(f"   Epochs Completed: {min(epochs)} to {max(epochs)} ({len(epochs)} total)")
        if epoch_data:
            total_steps = sum(ed.get('total_steps', 0) for ed in epoch_data.values())
            print(f"   Total Training Steps: {total_steps:,}")
        
        print(f"\n📉 TRAIN LOSS")
        print(f"   Initial (Epoch {epochs[0]}): {train_losses[0]:.4f}")
        print(f"   Final (Epoch {epochs[-1]}):   {train_losses[-1]:.4f}")
        print(f"   Best:                         {min(train_losses):.4f}")
        train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"   Improvement:                  {train_improvement:.1f}%")
        
        print(f"\n📉 VALIDATION LOSS")
        print(f"   Initial (Epoch {epochs[0]}): {val_losses[0]:.4f}")
        print(f"   Final (Epoch {epochs[-1]}):   {val_losses[-1]:.4f}")
        best_idx = np.argmin(val_losses)
        print(f"   Best:                         {val_losses[best_idx]:.4f} (Epoch {epochs[best_idx]})")
        val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
        print(f"   Improvement:                  {val_improvement:.1f}%")
        
        print(f"\n🎯 CTC LOSS (Audio-Text Alignment)")
        print(f"   Train CTC:")
        print(f"     Initial: {train_ctc[0]:.4f}")
        print(f"     Final:   {train_ctc[-1]:.4f}")
        print(f"     Best:    {min(train_ctc):.4f}")
        print(f"   Val CTC:")
        print(f"     Initial: {val_ctc[0]:.4f}")
        print(f"     Final:   {val_ctc[-1]:.4f}")
        print(f"     Best:    {min(val_ctc):.4f}")
        
        print(f"\n🔍 OVERFITTING ANALYSIS")
        final_gap = val_losses[-1] - train_losses[-1]
        print(f"   Final Gap (Val - Train): {final_gap:.4f}")
        if final_gap < 1.0:
            status = "✅ Excellent - No overfitting"
        elif final_gap < 2.0:
            status = "🟡 Good - Slight overfitting"
        elif final_gap < 3.0:
            status = "🟠 Moderate overfitting"
        else:
            status = "⚠️  Significant overfitting"
        print(f"   Status: {status}")
        
        # Check for convergence
        if len(val_losses) >= 3:
            recent_change = abs(val_losses[-1] - val_losses[-2])
            print(f"\n📈 CONVERGENCE")
            print(f"   Last epoch change: {recent_change:.4f}")
            if recent_change < 0.01:
                print(f"   Status: ✅ Converged (change < 0.01)")
            elif recent_change < 0.05:
                print(f"   Status: 🟡 Nearly converged (change < 0.05)")
            else:
                print(f"   Status: 🔄 Still improving")
    
    if epoch_data:
        print(f"\n📊 WITHIN-EPOCH STATISTICS")
        for epoch_num in sorted(epoch_data.keys())[:3]:  # Show first 3 epochs
            ed = epoch_data[epoch_num]
            if ed['train_losses']:
                print(f"   Epoch {epoch_num}:")
                print(f"     Steps: {len(ed['steps'])}")
                print(f"     Loss range: {min(ed['train_losses']):.4f} - {max(ed['train_losses']):.4f}")
                print(f"     CTC range:  {min(ed['train_ctc']):.4f} - {max(ed['train_ctc']):.4f}")
    
    print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse raw Kaggle training logs and generate comprehensive graphs'
    )
    parser.add_argument('--log_file', type=str, required=True,
                        help='Text file containing raw Kaggle training logs')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save generated graphs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PARSING RAW KAGGLE TRAINING LOGS")
    print("="*80)
    print(f"Log file: {args.log_file}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Parse logs
    data = parse_raw_kaggle_logs(args.log_file)
    
    if data is None:
        print("❌ Failed to parse logs")
        return
    
    # Generate comprehensive plots
    plot_comprehensive_training_curves(data, args.output_dir)
    
    print("\n✅ Done! Check the output directory for graphs and analysis.")


if __name__ == "__main__":
    main()
