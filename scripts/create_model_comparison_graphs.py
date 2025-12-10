#!/usr/bin/env python3
"""
Create comprehensive comparison graphs for the 2 ASR models
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

def extract_checkpoint_data(checkpoint_dir):
    """Extract training metrics from all checkpoints"""
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_path.glob('checkpoint_epoch_*.pt'))
    
    data = {
        'epochs': [],
        'val_loss': [],
        'train_loss': []
    }
    
    print(f"Extracting data from {len(checkpoints)} checkpoints...")
    
    for ckpt_file in checkpoints:
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            val_loss = checkpoint.get('val_loss', None)
            train_loss = checkpoint.get('train_loss', None)
            
            if epoch and val_loss is not None:
                data['epochs'].append(epoch)
                data['val_loss'].append(val_loss)
                if train_loss is not None:
                    data['train_loss'].append(train_loss)
        except Exception as e:
            print(f"  Skipped {ckpt_file.name}: {e}")
    
    return data

def create_comparison_graphs():
    """Create comprehensive comparison graphs"""
    
    # Extract data from both model locations
    print("="*70)
    print("EXTRACTING MODEL DATA")
    print("="*70)
    
    print("\nModel 1: kaggle_asr_outputs/checkpoints")
    model1_data = extract_checkpoint_data('kaggle_asr_outputs/checkpoints')
    
    print("\nModel 2: checkpoints")
    model2_data = extract_checkpoint_data('checkpoints')
    
    # If model2 is empty, use archives
    if not model2_data['epochs']:
        print("\nModel 2: archives/checkpoints_backup")
        model2_data = extract_checkpoint_data('archives/checkpoints_backup')
    
    print(f"\nModel 1: {len(model1_data['epochs'])} epochs")
    print(f"Model 2: {len(model2_data['epochs'])} epochs")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ============================================================
    # 1. Validation Loss Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    if model1_data['epochs']:
        ax1.plot(model1_data['epochs'], model1_data['val_loss'], 
                marker='o', linewidth=2, markersize=6, 
                label='Model 1 (kaggle_asr_outputs)', color=colors[0])
    
    if model2_data['epochs']:
        ax1.plot(model2_data['epochs'], model2_data['val_loss'], 
                marker='s', linewidth=2, markersize=6,
                label='Model 2 (checkpoints)', color=colors[1])
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Loss Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============================================================
    # 2. Training Summary Box
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    summary_text = "TRAINING SUMMARY\n" + "="*30 + "\n\n"
    
    if model1_data['epochs']:
        min_loss1 = min(model1_data['val_loss'])
        best_epoch1 = model1_data['epochs'][model1_data['val_loss'].index(min_loss1)]
        summary_text += f"MODEL 1:\n"
        summary_text += f"  Epochs: {len(model1_data['epochs'])}\n"
        summary_text += f"  Best Val Loss: {min_loss1:.4f}\n"
        summary_text += f"  Best Epoch: {best_epoch1}\n"
        summary_text += f"  Final Loss: {model1_data['val_loss'][-1]:.4f}\n\n"
    
    if model2_data['epochs']:
        min_loss2 = min(model2_data['val_loss'])
        best_epoch2 = model2_data['epochs'][model2_data['val_loss'].index(min_loss2)]
        summary_text += f"MODEL 2:\n"
        summary_text += f"  Epochs: {len(model2_data['epochs'])}\n"
        summary_text += f"  Best Val Loss: {min_loss2:.4f}\n"
        summary_text += f"  Best Epoch: {best_epoch2}\n"
        summary_text += f"  Final Loss: {model2_data['val_loss'][-1]:.4f}\n"
    
    ax2.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # ============================================================
    # 3. Loss Improvement Rate
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    if model1_data['epochs'] and len(model1_data['val_loss']) > 1:
        improvements1 = np.diff(model1_data['val_loss'])
        ax3.plot(model1_data['epochs'][1:], improvements1, 
                marker='o', linewidth=2, label='Model 1', color=colors[0])
    
    if model2_data['epochs'] and len(model2_data['val_loss']) > 1:
        improvements2 = np.diff(model2_data['val_loss'])
        ax3.plot(model2_data['epochs'][1:], improvements2, 
                marker='s', linewidth=2, label='Model 2', color=colors[1])
    
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loss Change', fontsize=11, fontweight='bold')
    ax3.set_title('Loss Improvement Rate\n(Negative = Better)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # 4. Best Loss Comparison Bar Chart
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    models = []
    best_losses = []
    
    if model1_data['epochs']:
        models.append('Model 1')
        best_losses.append(min(model1_data['val_loss']))
    
    if model2_data['epochs']:
        models.append('Model 2')
        best_losses.append(min(model2_data['val_loss']))
    
    bars = ax4.bar(models, best_losses, color=[colors[0], colors[1]], alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, loss in zip(bars, best_losses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Best Validation Loss', fontsize=11, fontweight='bold')
    ax4.set_title('Best Loss Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # 5. Loss Distribution (Box Plot)
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    loss_data = []
    labels = []
    
    if model1_data['epochs']:
        loss_data.append(model1_data['val_loss'])
        labels.append('Model 1')
    
    if model2_data['epochs']:
        loss_data.append(model2_data['val_loss'])
        labels.append('Model 2')
    
    bp = ax5.boxplot(loss_data, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor=colors[0], alpha=0.5),
                     medianprops=dict(color='red', linewidth=2))
    
    # Color boxes differently
    for patch, color in zip(bp['boxes'], [colors[0], colors[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax5.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
    ax5.set_title('Loss Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # 6. Epoch-by-Epoch Comparison Table
    # ============================================================
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create comparison table
    table_data = []
    table_data.append(['Epoch', 'Model 1 Loss', 'Model 2 Loss', 'Difference', 'Better'])
    
    # Get common epochs
    if model1_data['epochs'] and model2_data['epochs']:
        common_epochs = sorted(set(model1_data['epochs']) & set(model2_data['epochs']))[:10]
        
        for epoch in common_epochs:
            idx1 = model1_data['epochs'].index(epoch)
            idx2 = model2_data['epochs'].index(epoch)
            
            loss1 = model1_data['val_loss'][idx1]
            loss2 = model2_data['val_loss'][idx2]
            diff = loss1 - loss2
            better = 'Model 1' if diff > 0 else 'Model 2'
            
            table_data.append([
                f'{epoch}',
                f'{loss1:.4f}',
                f'{loss2:.4f}',
                f'{abs(diff):.4f}',
                better
            ])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on better model
    for i in range(1, len(table_data)):
        if len(table_data[i]) > 4:
            if table_data[i][4] == 'Model 1':
                for j in range(5):
                    table[(i, j)].set_facecolor('#E3F2FD')
            else:
                for j in range(5):
                    table[(i, j)].set_facecolor('#FCE4EC')
    
    ax6.set_title('Epoch-by-Epoch Comparison (First 10 Common Epochs)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # ============================================================
    # Main title
    # ============================================================
    fig.suptitle('KonkaniVani ASR - Model Comparison Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = 'outputs/model_comparison_graphs.png'
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphs saved to: {output_file}")
    
    # Also save as PDF
    output_pdf = 'outputs/model_comparison_graphs.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ PDF saved to: {output_pdf}")
    
    plt.show()
    
    return model1_data, model2_data

def print_detailed_comparison(model1_data, model2_data):
    """Print detailed text comparison"""
    print("\n" + "="*70)
    print("DETAILED MODEL COMPARISON")
    print("="*70)
    
    if model1_data['epochs']:
        print("\nMODEL 1 (kaggle_asr_outputs):")
        print(f"  Total epochs: {len(model1_data['epochs'])}")
        print(f"  Epoch range: {min(model1_data['epochs'])} - {max(model1_data['epochs'])}")
        print(f"  Best val loss: {min(model1_data['val_loss']):.4f} (epoch {model1_data['epochs'][model1_data['val_loss'].index(min(model1_data['val_loss']))]})")
        print(f"  Final val loss: {model1_data['val_loss'][-1]:.4f}")
        print(f"  Loss improvement: {model1_data['val_loss'][0] - model1_data['val_loss'][-1]:.4f}")
    
    if model2_data['epochs']:
        print("\nMODEL 2 (checkpoints):")
        print(f"  Total epochs: {len(model2_data['epochs'])}")
        print(f"  Epoch range: {min(model2_data['epochs'])} - {max(model2_data['epochs'])}")
        print(f"  Best val loss: {min(model2_data['val_loss']):.4f} (epoch {model2_data['epochs'][model2_data['val_loss'].index(min(model2_data['val_loss']))]})")
        print(f"  Final val loss: {model2_data['val_loss'][-1]:.4f}")
        print(f"  Loss improvement: {model2_data['val_loss'][0] - model2_data['val_loss'][-1]:.4f}")
    
    if model1_data['epochs'] and model2_data['epochs']:
        print("\nCOMPARISON:")
        best1 = min(model1_data['val_loss'])
        best2 = min(model2_data['val_loss'])
        
        if best1 < best2:
            print(f"  ✓ Model 1 is better by {best2 - best1:.4f} loss")
        else:
            print(f"  ✓ Model 2 is better by {best1 - best2:.4f} loss")

if __name__ == '__main__':
    model1_data, model2_data = create_comparison_graphs()
    print_detailed_comparison(model1_data, model2_data)
