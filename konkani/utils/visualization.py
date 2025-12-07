"""
Visualization utilities for Konkani NLP.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None
):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[Path] = None,
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: Names of the classes
        save_path: Path to save the plot
        title: Title for the plot
    """
    if class_names is None:
        class_names = ['negative', 'neutral', 'positive']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_label_distribution(
    labels: List[int],
    class_names: List[str] = None,
    save_path: Optional[Path] = None,
    title: str = 'Label Distribution'
):
    """
    Plot label distribution.
    
    Args:
        labels: List of label indices
        class_names: Names of the classes
        save_path: Path to save the plot
        title: Title for the plot
    """
    if class_names is None:
        class_names = ['negative', 'neutral', 'positive']
    
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar([class_names[i] for i in unique], counts, color='skyblue', edgecolor='navy')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (label, count) in enumerate(zip(unique, counts)):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Label distribution saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
