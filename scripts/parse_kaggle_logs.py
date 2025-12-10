"""
Parse Kaggle Training Logs and Generate Graphs
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_kaggle_logs(log_file):
    """Parse Kaggle training logs from text file"""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract epoch summaries
    epoch_pattern = r'Epoch (\d+)/\d+\s+Train Loss: ([\d.]+) \(CTC: ([\d.]+)\)\s+Val Loss: ([\d.]+) \(CTC: ([\d.]+)\)'
    
    epochs = []
    train_losses = []
    train_ctc_losses = []
    val_losses = []
    val_ctc_losses = []
    
    for match in re.finditer(epoch_pattern, content):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        train_ctc = float(match.group(3))
        val_loss = float(match.group(4))
        val_ctc = float(match.group(5))
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        train_ctc_losses.append(train_ctc)
        val_losses.append(val_loss)
        val_ctc_losses.append(val_ctc)
    
    if not epochs:
        print("❌ No epoch data found in logs")
        return None
    
    print(f"✅ Found {len(epochs)} epochs of training data")
    print(f"   Epochs: {min(epochs)} to {max(epochs)}")
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_ctc': train_ctc_losses,
        'val_loss': val_losses,
        'val_ctc': val_ctc_losses
    }


def plot_training_curves(data, output_dir='outputs'):
    """Generate training curve plots"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = data['epochs']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('KonkaniVani ASR Training Curves (Kaggle)', fontsize=16, fontweight='bold')
    
    # Plot 1: Train vs Val Loss
    ax = axes[0, 0]
    ax.plot(epochs, data['train_loss'], label='Train Loss', linewidth=2, 
            color='#2E86AB', marker='o', markersize=4)
    ax.plot(epochs, data['val_loss'], label='Val Loss', linewidth=2, 
            color='#A23B72', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: CTC Loss
    ax = axes[0, 1]
    ax.plot(epochs, data['train_ctc'], label='Train CTC Loss', linewidth=2, 
            color='#2E86AB', marker='o', markersize=4)
    ax.plot(epochs, data['val_ctc'], label='Val CTC Loss', linewidth=2, 
            color='#A23B72', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('CTC Loss', fontsize=12)
    ax.set_title('CTC Loss (Audio-Text Alignment)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Val Loss Detailed
    ax = axes[1, 0]
    ax.plot(epochs, data['val_loss'], linewidth=2, color='#A23B72', 
            marker='o', markersize=5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Loss', fontsize=12)
    ax.set_title('Validation Loss Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight best epoch
    best_idx = np.argmin(data['val_loss'])
    best_epoch = epochs[best_idx]
    best_loss = data['val_loss'][best_idx]
    ax.plot(best_epoch, best_loss, 'r*', markersize=15, label=f'Best: Epoch {best_epoch}')
    ax.legend(fontsize=10)
    
    # Plot 4: Loss Improvement
    ax = axes[1, 1]
    train_improvement = [(data['train_loss'][0] - loss) / data['train_loss'][0] * 100 
                         for loss in data['train_loss']]
    val_improvement = [(data['val_loss'][0] - loss) / data['val_loss'][0] * 100 
                       for loss in data['val_loss']]
    
    ax.plot(epochs, train_improvement, label='Train Improvement', linewidth=2, 
            color='#2E86AB', marker='o', markersize=4)
    ax.plot(epochs, val_improvement, label='Val Improvement', linewidth=2, 
            color='#A23B72', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Loss Improvement from Epoch 1', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'kaggle_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved training curves to: {output_path}")
    
    # Also save as PDF
    pdf_path = Path(output_dir) / 'kaggle_training_curves.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved PDF version to: {pdf_path}")
    
    plt.close()
    
    # Print summary
    print_summary(data)


def print_summary(data):
    """Print training summary statistics"""
    epochs = data['epochs']
    
    print("\n" + "="*70)
    print("KAGGLE TRAINING SUMMARY")
    print("="*70)
    
    print(f"\nEpochs Trained: {min(epochs)} to {max(epochs)} ({len(epochs)} total)")
    
    print(f"\nTrain Loss:")
    print(f"  Initial (Epoch {epochs[0]}): {data['train_loss'][0]:.4f}")
    print(f"  Final (Epoch {epochs[-1]}):   {data['train_loss'][-1]:.4f}")
    print(f"  Best:    {min(data['train_loss']):.4f}")
    improvement = (data['train_loss'][0] - data['train_loss'][-1]) / data['train_loss'][0] * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    print(f"\nValidation Loss:")
    print(f"  Initial (Epoch {epochs[0]}): {data['val_loss'][0]:.4f}")
    print(f"  Final (Epoch {epochs[-1]}):   {data['val_loss'][-1]:.4f}")
    best_idx = np.argmin(data['val_loss'])
    print(f"  Best:    {data['val_loss'][best_idx]:.4f} (Epoch {epochs[best_idx]})")
    improvement = (data['val_loss'][0] - data['val_loss'][-1]) / data['val_loss'][0] * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    print(f"\nCTC Loss (Audio-Text Alignment):")
    print(f"  Train CTC - Initial: {data['train_ctc'][0]:.4f}, Final: {data['train_ctc'][-1]:.4f}")
    print(f"  Val CTC   - Initial: {data['val_ctc'][0]:.4f}, Final: {data['val_ctc'][-1]:.4f}")
    
    # Check for overfitting
    final_gap = data['val_loss'][-1] - data['train_loss'][-1]
    print(f"\nOverfitting Check:")
    print(f"  Final gap (Val - Train): {final_gap:.4f}")
    if final_gap < 1.0:
        print(f"  Status: ✅ No overfitting (gap < 1.0)")
    elif final_gap < 2.0:
        print(f"  Status: 🟡 Slight overfitting (gap < 2.0)")
    else:
        print(f"  Status: ⚠️  Overfitting detected (gap > 2.0)")
    
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse Kaggle logs and generate training graphs')
    parser.add_argument('--log_file', type=str, default='download.txt',
                        help='Text file containing Kaggle training logs')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save generated graphs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PARSING KAGGLE TRAINING LOGS")
    print("="*70)
    print(f"Log file: {args.log_file}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Parse logs
    data = parse_kaggle_logs(args.log_file)
    
    if data is None:
        print("❌ Failed to parse logs")
        return
    
    # Generate plots
    plot_training_curves(data, args.output_dir)
    
    print("\n✅ Done! Check the output directory for graphs.")


if __name__ == "__main__":
    main()
