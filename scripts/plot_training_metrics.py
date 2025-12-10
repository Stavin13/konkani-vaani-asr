#!/usr/bin/env python3
"""
Extract and plot training metrics from checkpoint files.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def extract_metrics_from_checkpoints(checkpoint_dir):
    """Extract training metrics from all checkpoint files."""
    checkpoint_dir = Path(checkpoint_dir)
    
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_wer': [],
        'val_wer': [],
        'train_cer': [],
        'val_cer': [],
        'learning_rate': []
    }
    
    # Get all checkpoint files and sort by epoch number
    checkpoint_files = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pt'),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    for ckpt_file in checkpoint_files:
        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            
            print(f"Processing epoch {epoch}...")
            
            metrics['epoch'].append(epoch)
            metrics['train_loss'].append(checkpoint.get('train_loss', None))
            metrics['val_loss'].append(checkpoint.get('val_loss', None))
            metrics['train_wer'].append(checkpoint.get('train_wer', None))
            metrics['val_wer'].append(checkpoint.get('val_wer', None))
            metrics['train_cer'].append(checkpoint.get('train_cer', None))
            metrics['val_cer'].append(checkpoint.get('val_cer', None))
            
            # Extract learning rate from optimizer if available
            if 'optimizer' in checkpoint:
                lr = checkpoint['optimizer']['param_groups'][0]['lr']
                metrics['learning_rate'].append(lr)
            else:
                metrics['learning_rate'].append(None)
                
        except Exception as e:
            print(f"Error loading {ckpt_file}: {e}")
            continue
    
    return metrics


def plot_metrics(metrics, output_dir='outputs'):
    """Create comprehensive training plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    epochs = metrics['epoch']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ASR Training Metrics - 50 Epochs', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    if any(metrics['train_loss']):
        ax1.plot(epochs, metrics['train_loss'], 'b-o', label='Train Loss', markersize=4)
    if any(metrics['val_loss']):
        ax1.plot(epochs, metrics['val_loss'], 'r-s', label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: WER (Word Error Rate)
    ax2 = axes[0, 1]
    if any(metrics['train_wer']):
        ax2.plot(epochs, metrics['train_wer'], 'b-o', label='Train WER', markersize=4)
    if any(metrics['val_wer']):
        ax2.plot(epochs, metrics['val_wer'], 'r-s', label='Val WER', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('WER (%)')
    ax2.set_title('Word Error Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CER (Character Error Rate)
    ax3 = axes[1, 0]
    if any(metrics['train_cer']):
        ax3.plot(epochs, metrics['train_cer'], 'b-o', label='Train CER', markersize=4)
    if any(metrics['val_cer']):
        ax3.plot(epochs, metrics['val_cer'], 'r-s', label='Val CER', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('CER (%)')
    ax3.set_title('Character Error Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    ax4 = axes[1, 1]
    if any(metrics['learning_rate']):
        ax4.plot(epochs, metrics['learning_rate'], 'g-o', markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'training_metrics_50epochs.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_file}")
    
    # Also create individual plots for better detail
    create_individual_plots(metrics, output_dir)
    
    plt.show()


def create_individual_plots(metrics, output_dir):
    """Create individual detailed plots."""
    epochs = metrics['epoch']
    
    # Loss plot
    if any(metrics['train_loss']) or any(metrics['val_loss']):
        plt.figure(figsize=(10, 6))
        if any(metrics['train_loss']):
            plt.plot(epochs, metrics['train_loss'], 'b-o', label='Train Loss', markersize=5, linewidth=2)
        if any(metrics['val_loss']):
            plt.plot(epochs, metrics['val_loss'], 'r-s', label='Val Loss', markersize=5, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Over 50 Epochs', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_plot.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'loss_plot.png'}")
        plt.close()
    
    # WER plot
    if any(metrics['train_wer']) or any(metrics['val_wer']):
        plt.figure(figsize=(10, 6))
        if any(metrics['train_wer']):
            plt.plot(epochs, metrics['train_wer'], 'b-o', label='Train WER', markersize=5, linewidth=2)
        if any(metrics['val_wer']):
            plt.plot(epochs, metrics['val_wer'], 'r-s', label='Val WER', markersize=5, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('WER (%)', fontsize=12)
        plt.title('Word Error Rate Over 50 Epochs', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'wer_plot.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'wer_plot.png'}")
        plt.close()


def print_summary(metrics):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if any(metrics['val_loss']):
        val_losses = [x for x in metrics['val_loss'] if x is not None]
        best_loss_idx = val_losses.index(min(val_losses))
        print(f"\nBest Validation Loss: {min(val_losses):.4f} at epoch {metrics['epoch'][best_loss_idx]}")
    
    if any(metrics['val_wer']):
        val_wers = [x for x in metrics['val_wer'] if x is not None]
        best_wer_idx = val_wers.index(min(val_wers))
        print(f"Best Validation WER: {min(val_wers):.2f}% at epoch {metrics['epoch'][best_wer_idx]}")
    
    if any(metrics['val_cer']):
        val_cers = [x for x in metrics['val_cer'] if x is not None]
        best_cer_idx = val_cers.index(min(val_cers))
        print(f"Best Validation CER: {min(val_cers):.2f}% at epoch {metrics['epoch'][best_cer_idx]}")
    
    print("\nFinal Epoch Metrics:")
    if metrics['train_loss'][-1]:
        print(f"  Train Loss: {metrics['train_loss'][-1]:.4f}")
    if metrics['val_loss'][-1]:
        print(f"  Val Loss: {metrics['val_loss'][-1]:.4f}")
    if metrics['train_wer'][-1]:
        print(f"  Train WER: {metrics['train_wer'][-1]:.2f}%")
    if metrics['val_wer'][-1]:
        print(f"  Val WER: {metrics['val_wer'][-1]:.2f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from checkpoints')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='kaggle_asr_outputs/checkpoints',
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    print("Extracting metrics from checkpoints...")
    metrics = extract_metrics_from_checkpoints(args.checkpoint_dir)
    
    if not metrics['epoch']:
        print("No metrics found in checkpoints!")
        return
    
    print(f"\nExtracted metrics for {len(metrics['epoch'])} epochs")
    
    print_summary(metrics)
    
    print("\nGenerating plots...")
    plot_metrics(metrics, args.output_dir)
    
    print("\n✓ Done! Check the outputs directory for your graphs.")


if __name__ == '__main__':
    main()
