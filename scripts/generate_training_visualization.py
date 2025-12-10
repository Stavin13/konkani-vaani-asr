#!/usr/bin/env python3
"""
Generate comprehensive training visualization graphs from Kaggle training logs.
Creates a multi-panel figure showing:
1. Training vs Validation Loss
2. Training vs Validation Accuracy
3. Final Loss Comparison
4. Training Summary Statistics
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
import numpy as np


def parse_training_log(log_file):
    """Parse training log file and extract metrics."""
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse epoch completion lines
            if 'Epoch' in line and 'completed' in line:
                try:
                    parts = line.split()
                    epoch_idx = parts.index('Epoch') + 1
                    epoch = int(parts[epoch_idx].rstrip(':'))
                    
                    # Extract train loss
                    if 'Train Loss:' in line:
                        train_loss_idx = parts.index('Loss:') + 1
                        train_loss = float(parts[train_loss_idx].rstrip(','))
                        train_losses.append(train_loss)
                    
                    # Extract val loss
                    if 'Val Loss:' in line:
                        val_loss_idx = parts.index('Loss:') + 1
                        val_loss = float(parts[val_loss_idx].rstrip(','))
                        val_losses.append(val_loss)
                    
                    # Extract train accuracy
                    if 'Train Acc:' in line:
                        train_acc_idx = parts.index('Acc:') + 1
                        train_acc = float(parts[train_acc_idx].rstrip('%,'))
                        train_accs.append(train_acc)
                    
                    # Extract val accuracy
                    if 'Val Acc:' in line:
                        val_acc_idx = parts.index('Acc:') + 1
                        val_acc = float(parts[val_acc_idx].rstrip('%,'))
                        val_accs.append(val_acc)
                        
                        epochs.append(epoch)
                        
                except (ValueError, IndexError) as e:
                    continue
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }


def create_training_visualization(data, output_path='training_metrics.png', title_prefix='Konkani ASR'):
    """Create comprehensive training visualization."""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{title_prefix} Training Metrics', fontsize=16, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    epochs = data['epochs']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    train_acc = data['train_acc']
    val_acc = data['val_acc']
    
    # 1. Training vs Validation Loss (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, 'o-', color='blue', linewidth=2, markersize=6, label='Training Loss')
    ax1.plot(epochs, val_loss, 's-', color='red', linewidth=2, markersize=6, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Training vs Validation Accuracy (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_acc, 'o-', color='green', linewidth=2, markersize=6, label='Training Accuracy')
    ax2.plot(epochs, val_acc, 's-', color='purple', linewidth=2, markersize=6, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Final Loss Comparison (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    final_train_loss = train_loss[-1] if train_loss else 0
    final_val_loss = val_loss[-1] if val_loss else 0
    
    bars = ax3.bar(['Training', 'Validation'], [final_train_loss, final_val_loss], 
                   color=['blue', 'red'], alpha=0.7, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel('Final Loss', fontsize=11)
    ax3.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, max(final_train_loss, final_val_loss) * 1.2)
    
    # 4. Training Summary (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary text box
    summary_text = f"""Training Summary

Total Epochs: {len(epochs)}

Training Loss: {train_loss[0]:.2f} → {train_loss[-1]:.2f}
Validation Loss: {val_loss[0]:.2f} → {val_loss[-1]:.2f}

Training Accuracy: {train_acc[0]:.0f}% → {train_acc[-1]:.0f}%
Validation Accuracy: {val_acc[0]:.0f}% → {val_acc[-1]:.0f}%

Best Val Loss: {min(val_loss):.2f} (Epoch {epochs[val_loss.index(min(val_loss))]})
Best Val Acc: {max(val_acc):.0f}% (Epoch {epochs[val_acc.index(max(val_acc))]})"""
    
    # Add text box with beige background
    bbox_props = dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=bbox_props, family='monospace')
    
    ax4.set_title('Training Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Training visualization saved to: {output_path}")
    
    # Also display if in interactive mode
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate training visualization from logs')
    parser.add_argument('--log', type=str, default='kaggle_training_log.txt',
                       help='Path to training log file')
    parser.add_argument('--output', type=str, default='training_metrics.png',
                       help='Output image path')
    parser.add_argument('--title', type=str, default='Konkani ASR',
                       help='Title prefix for the visualization')
    
    args = parser.parse_args()
    
    log_path = Path(args.log)
    
    if not log_path.exists():
        print(f"✗ Log file not found: {log_path}")
        print("\nSearching for log files...")
        
        # Search common locations
        search_paths = [
            Path('kaggle_training_log.txt'),
            Path('logs/training.log'),
            Path('/kaggle/working/logs/training.log'),
            Path('checkpoints/training.log')
        ]
        
        for path in search_paths:
            if path.exists():
                print(f"✓ Found log file: {path}")
                log_path = path
                break
        else:
            print("✗ No log files found. Please specify --log path")
            return
    
    print(f"Parsing training log: {log_path}")
    data = parse_training_log(log_path)
    
    if not data['epochs']:
        print("✗ No training data found in log file")
        return
    
    print(f"✓ Found {len(data['epochs'])} epochs of training data")
    print(f"  Train Loss: {data['train_loss'][0]:.3f} → {data['train_loss'][-1]:.3f}")
    print(f"  Val Loss: {data['val_loss'][0]:.3f} → {data['val_loss'][-1]:.3f}")
    print(f"  Train Acc: {data['train_acc'][0]:.1f}% → {data['train_acc'][-1]:.1f}%")
    print(f"  Val Acc: {data['val_acc'][0]:.1f}% → {data['val_acc'][-1]:.1f}%")
    
    create_training_visualization(data, args.output, args.title)


if __name__ == '__main__':
    main()
