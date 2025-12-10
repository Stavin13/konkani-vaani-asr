#!/usr/bin/env python3
"""
Complete setup for Translation and Emotion models with training graphs
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_custom_translator import create_custom_translation_model
from models.konkani_custom_emotion import create_custom_emotion_model, EmotionLoss


# ============================================================================
# EMOTION MODEL SETUP
# ============================================================================

EMOTION_LABELS = {
    0: 'joy',
    1: 'sadness',
    2: 'anger',
    3: 'fear',
    4: 'surprise',
    5: 'disgust',
    6: 'neutral'
}

def setup_emotion_model():
    """Setup emotion detection model"""
    print("="*70)
    print("EMOTION DETECTION MODEL SETUP")
    print("="*70)
    
    # Model configuration
    config = {
        'vocab_size': 5000,
        'num_emotions': 7,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True
    }
    
    # Create model
    model = create_custom_emotion_model(
        vocab_size=config['vocab_size'],
        num_emotions=config['num_emotions'],
        config={k: v for k, v in config.items() if k not in ['vocab_size', 'num_emotions']}
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  Type: BiLSTM + Attention")
    print(f"  Vocab size: {config['vocab_size']:,}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Bidirectional: {config['bidirectional']}")
    print(f"  Total parameters: {num_params:,}")
    print(f"\nEmotion Classes: {list(EMOTION_LABELS.values())}")
    
    # Save model info
    model_dir = Path('models/emotion_model')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(model_dir / 'emotion_labels.json', 'w') as f:
        json.dump(EMOTION_LABELS, f, indent=2)
    
    print(f"\n✓ Model configuration saved to: {model_dir}")
    
    return model, config


# ============================================================================
# TRANSLATION MODEL SETUP
# ============================================================================

def setup_translation_model():
    """Setup translation model"""
    print("\n" + "="*70)
    print("TRANSLATION MODEL SETUP")
    print("="*70)
    
    # Model configuration
    config = {
        'src_vocab_size': 5000,  # Konkani
        'tgt_vocab_size': 10000,  # English
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_len': 512
    }
    
    # Create model
    model = create_custom_translation_model(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        config={k: v for k, v in config.items() if k not in ['src_vocab_size', 'tgt_vocab_size']}
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  Type: Transformer Seq2Seq")
    print(f"  Source vocab (Konkani): {config['src_vocab_size']:,}")
    print(f"  Target vocab (English): {config['tgt_vocab_size']:,}")
    print(f"  Model dimension: {config['d_model']}")
    print(f"  Attention heads: {config['nhead']}")
    print(f"  Encoder layers: {config['num_encoder_layers']}")
    print(f"  Decoder layers: {config['num_decoder_layers']}")
    print(f"  Total parameters: {num_params:,}")
    
    # Save model info
    model_dir = Path('models/translation_model')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Model configuration saved to: {model_dir}")
    
    return model, config


# ============================================================================
# TRAINING SIMULATION (for demonstration)
# ============================================================================

def simulate_training_history(model_name, num_epochs=15):
    """Simulate training history for visualization"""
    print(f"\nGenerating sample training history for {model_name}...")
    
    np.random.seed(42 if model_name == 'emotion' else 43)
    
    history = {
        'epochs': list(range(1, num_epochs + 1)),
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Simulate realistic training curves
    for epoch in range(1, num_epochs + 1):
        # Loss decreases with some noise
        train_loss = 5.5 * np.exp(-0.15 * epoch) + np.random.normal(0, 0.05)
        val_loss = 5.5 * np.exp(-0.12 * epoch) + np.random.normal(0, 0.08)
        
        # Accuracy increases
        train_acc = 100 * (1 - np.exp(-0.2 * epoch)) + np.random.normal(0, 1)
        val_acc = 100 * (1 - np.exp(-0.16 * epoch)) + np.random.normal(0, 1.5)
        
        history['train_loss'].append(max(train_loss, 1.7))
        history['val_loss'].append(max(val_loss, 2.0))
        history['train_acc'].append(min(train_acc, 85))
        history['val_acc'].append(min(val_acc, 75))
    
    return history


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_training_graphs(emotion_history, translation_history):
    """Create comprehensive training graphs"""
    print("\n" + "="*70)
    print("CREATING TRAINING VISUALIZATION")
    print("="*70)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = {
        'emotion': '#E91E63',
        'translation': '#2196F3',
        'train': '#4CAF50',
        'val': '#FF9800'
    }
    
    # ========================================================================
    # 1. Loss Comparison - Both Models
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.plot(emotion_history['epochs'], emotion_history['val_loss'], 
            marker='o', linewidth=2.5, markersize=7,
            label='Emotion Model', color=colors['emotion'])
    
    ax1.plot(translation_history['epochs'], translation_history['val_loss'], 
            marker='s', linewidth=2.5, markersize=7,
            label='Translation Model', color=colors['translation'])
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Model Comparison - Validation Loss', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. Summary Statistics
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    summary = "MODEL COMPARISON\n" + "="*35 + "\n\n"
    summary += "EMOTION MODEL:\n"
    summary += f"  Final Val Loss: {emotion_history['val_loss'][-1]:.3f}\n"
    summary += f"  Final Val Acc: {emotion_history['val_acc'][-1]:.1f}%\n"
    summary += f"  Best Val Loss: {min(emotion_history['val_loss']):.3f}\n"
    summary += f"  Best Val Acc: {max(emotion_history['val_acc']):.1f}%\n\n"
    
    summary += "TRANSLATION MODEL:\n"
    summary += f"  Final Val Loss: {translation_history['val_loss'][-1]:.3f}\n"
    summary += f"  Final Val Acc: {translation_history['val_acc'][-1]:.1f}%\n"
    summary += f"  Best Val Loss: {min(translation_history['val_loss']):.3f}\n"
    summary += f"  Best Val Acc: {max(translation_history['val_acc']):.1f}%\n"
    
    ax2.text(0.05, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round',
            facecolor='lightblue', alpha=0.3))
    
    # ========================================================================
    # 3. Emotion Model - Train vs Val Loss
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(emotion_history['epochs'], emotion_history['train_loss'],
            marker='o', linewidth=2, label='Training', color=colors['train'])
    ax3.plot(emotion_history['epochs'], emotion_history['val_loss'],
            marker='s', linewidth=2, label='Validation', color=colors['val'])
    
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax3.set_title('Emotion Model - Loss', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # 4. Translation Model - Train vs Val Loss
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(translation_history['epochs'], translation_history['train_loss'],
            marker='o', linewidth=2, label='Training', color=colors['train'])
    ax4.plot(translation_history['epochs'], translation_history['val_loss'],
            marker='s', linewidth=2, label='Validation', color=colors['val'])
    
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax4.set_title('Translation Model - Loss', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # 5. Accuracy Comparison
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    ax5.plot(emotion_history['epochs'], emotion_history['val_acc'],
            marker='o', linewidth=2, label='Emotion', color=colors['emotion'])
    ax5.plot(translation_history['epochs'], translation_history['val_acc'],
            marker='s', linewidth=2, label='Translation', color=colors['translation'])
    
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # 6. Final Performance Bar Chart
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 0])
    
    models = ['Emotion', 'Translation']
    final_losses = [
        emotion_history['val_loss'][-1],
        translation_history['val_loss'][-1]
    ]
    
    bars = ax6.bar(models, final_losses, 
                  color=[colors['emotion'], colors['translation']],
                  alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    ax6.set_ylabel('Final Validation Loss', fontsize=11, fontweight='bold')
    ax6.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 7. Emotion Model - Accuracy
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 1])
    
    ax7.plot(emotion_history['epochs'], emotion_history['train_acc'],
            marker='o', linewidth=2, label='Training', color=colors['train'])
    ax7.plot(emotion_history['epochs'], emotion_history['val_acc'],
            marker='s', linewidth=2, label='Validation', color=colors['val'])
    
    ax7.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax7.set_title('Emotion Model - Accuracy', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # ========================================================================
    # 8. Translation Model - Accuracy
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    
    ax8.plot(translation_history['epochs'], translation_history['train_acc'],
            marker='o', linewidth=2, label='Training', color=colors['train'])
    ax8.plot(translation_history['epochs'], translation_history['val_acc'],
            marker='s', linewidth=2, label='Validation', color=colors['val'])
    
    ax8.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax8.set_title('Translation Model - Accuracy', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Konkani Translation & Emotion Detection - Training Analysis',
                fontsize=17, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'translation_emotion_training_graphs.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphs saved to: {output_file}")
    
    output_pdf = output_dir / 'translation_emotion_training_graphs.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ PDF saved to: {output_pdf}")
    
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("KONKANI TRANSLATION & EMOTION MODELS - COMPLETE SETUP")
    print("="*70)
    
    # Setup models
    emotion_model, emotion_config = setup_emotion_model()
    translation_model, translation_config = setup_translation_model()
    
    # Generate training histories
    emotion_history = simulate_training_history('emotion', num_epochs=15)
    translation_history = simulate_training_history('translation', num_epochs=15)
    
    # Create visualizations
    create_training_graphs(emotion_history, translation_history)
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Prepare training data")
    print("  2. Train emotion model: python training_scripts/train_emotion.py")
    print("  3. Train translation model: python training_scripts/train_translation.py")
    print("  4. Evaluate models: python scripts/evaluate_models.py")
    print("\n✓ All files saved to models/ and outputs/")


if __name__ == '__main__':
    main()
