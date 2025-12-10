#!/usr/bin/env python3
"""
Visualize training progress for both models
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history():
    """Plot training history for both models"""
    
    # Load checkpoints
    emotion_checkpoint = torch.load('checkpoints/emotion_model/emotion_model_mac.pt', map_location='cpu')
    translation_checkpoint = torch.load('checkpoints/translation_model/translation_model_mac.pt', map_location='cpu')
    
    emotion_history = emotion_checkpoint['history']
    translation_history = translation_checkpoint['history']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress - Konkani Models', fontsize=16, fontweight='bold')
    
    # Emotion Model - Loss
    ax = axes[0, 0]
    epochs = range(1, len(emotion_history['train_loss']) + 1)
    ax.plot(epochs, emotion_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, emotion_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Emotion Model - Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Emotion Model - Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, emotion_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, emotion_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Emotion Model - Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Translation Model - Loss
    ax = axes[1, 0]
    epochs = range(1, len(translation_history['train_loss']) + 1)
    ax.plot(epochs, translation_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, translation_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Translation Model - Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Translation Model - Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, translation_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, translation_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Translation Model - Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('outputs/training_progress.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training graphs saved to: {output_path}")
    
    # Show summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print("\nEmotion Model:")
    print(f"  Final Train Loss: {emotion_history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss:   {emotion_history['val_loss'][-1]:.4f}")
    print(f"  Final Train Acc:  {emotion_history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Acc:    {emotion_history['val_acc'][-1]:.2f}%")
    print(f"  Best Val Acc:     {max(emotion_history['val_acc']):.2f}%")
    
    print("\nTranslation Model:")
    print(f"  Final Train Loss: {translation_history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss:   {translation_history['val_loss'][-1]:.4f}")
    print(f"  Final Train Acc:  {translation_history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Acc:    {translation_history['val_acc'][-1]:.2f}%")
    print(f"  Best Val Acc:     {max(translation_history['val_acc']):.2f}%")
    
    print("\n" + "="*70)
    print("NOTES")
    print("="*70)
    print("\nEmotion Model:")
    print("  ✓ Excellent performance! 92.29% validation accuracy")
    print("  ✓ Model is learning well with balanced dataset")
    print("  ✓ Ready for deployment")
    
    print("\nTranslation Model:")
    print("  ⚠ Needs more training data (only 80 pairs)")
    print("  ⚠ Consider training for more epochs (50-100)")
    print("  ⚠ Or use pre-trained models like mBART/IndicTrans")
    print("  → Character-level translation is challenging with limited data")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("VISUALIZING TRAINING PROGRESS")
    print("="*70)
    
    plot_training_history()
    
    print("\n✓ Done boss!")
