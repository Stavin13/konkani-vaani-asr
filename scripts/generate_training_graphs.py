"""
Generate Training Graphs from TensorBoard Logs
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not installed. Install with: pip install tensorboard")


def read_tensorboard_logs(log_dir):
    """Read TensorBoard event files and extract metrics"""
    if not TENSORBOARD_AVAILABLE:
        return None
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
    
    print(f"Found {len(event_files)} event files")
    
    # Read the most recent/largest event file
    event_file = max(event_files, key=os.path.getsize)
    print(f"Reading: {os.path.basename(event_file)}")
    
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    print(f"Available metrics: {tags.get('scalars', [])}")
    
    metrics = {}
    
    # Extract common metrics
    for tag in ['train/loss', 'train/ctc_loss', 'val/loss', 'val/ctc_loss', 'train/lr']:
        if tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            metrics[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
    
    return metrics


def plot_training_curves(metrics, output_dir='outputs'):
    """Generate training curve plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('KonkaniVani ASR Training Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Train vs Val Loss
    ax = axes[0, 0]
    if 'train/loss' in metrics:
        ax.plot(metrics['train/loss']['steps'], metrics['train/loss']['values'], 
                label='Train Loss', linewidth=2, color='#2E86AB')
    if 'val/loss' in metrics:
        ax.plot(metrics['val/loss']['steps'], metrics['val/loss']['values'], 
                label='Val Loss', linewidth=2, color='#A23B72')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: CTC Loss
    ax = axes[0, 1]
    if 'train/ctc_loss' in metrics:
        ax.plot(metrics['train/ctc_loss']['steps'], metrics['train/ctc_loss']['values'], 
                label='Train CTC Loss', linewidth=2, color='#2E86AB')
    if 'val/ctc_loss' in metrics:
        ax.plot(metrics['val/ctc_loss']['steps'], metrics['val/ctc_loss']['values'], 
                label='Val CTC Loss', linewidth=2, color='#A23B72')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('CTC Loss', fontsize=12)
    ax.set_title('CTC Loss (Audio-Text Alignment)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax = axes[1, 0]
    if 'train/lr' in metrics:
        ax.plot(metrics['train/lr']['steps'], metrics['train/lr']['values'], 
                linewidth=2, color='#F18F01')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Plot 4: Val Loss Zoomed (last 50%)
    ax = axes[1, 1]
    if 'val/loss' in metrics:
        steps = metrics['val/loss']['steps']
        values = metrics['val/loss']['values']
        mid_point = len(steps) // 2
        ax.plot(steps[mid_point:], values[mid_point:], 
                linewidth=2, color='#A23B72', marker='o', markersize=4)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Val Loss', fontsize=12)
        ax.set_title('Validation Loss (Last 50% of Training)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved training curves to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, 'training_curves.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved PDF version to: {pdf_path}")
    
    plt.close()
    
    # Create summary statistics
    print_summary(metrics)


def print_summary(metrics):
    """Print training summary statistics"""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    if 'train/loss' in metrics:
        train_loss = metrics['train/loss']['values']
        print(f"\nTrain Loss:")
        print(f"  Initial: {train_loss[0]:.4f}")
        print(f"  Final:   {train_loss[-1]:.4f}")
        print(f"  Best:    {min(train_loss):.4f}")
        print(f"  Improvement: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%")
    
    if 'val/loss' in metrics:
        val_loss = metrics['val/loss']['values']
        print(f"\nValidation Loss:")
        print(f"  Initial: {val_loss[0]:.4f}")
        print(f"  Final:   {val_loss[-1]:.4f}")
        print(f"  Best:    {min(val_loss):.4f}")
        print(f"  Improvement: {((val_loss[0] - val_loss[-1]) / val_loss[0] * 100):.1f}%")
    
    if 'train/lr' in metrics:
        lr = metrics['train/lr']['values']
        print(f"\nLearning Rate:")
        print(f"  Initial: {lr[0]:.6f}")
        print(f"  Final:   {lr[-1]:.6f}")
    
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training graphs from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Directory containing TensorBoard logs')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save generated graphs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATING TRAINING GRAPHS")
    print("="*70)
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Read logs
    metrics = read_tensorboard_logs(args.log_dir)
    
    if metrics is None or len(metrics) == 0:
        print("❌ No metrics found in logs")
        return
    
    # Generate plots
    plot_training_curves(metrics, args.output_dir)
    
    print("\n✅ Done! Check the output directory for graphs.")


if __name__ == "__main__":
    main()
