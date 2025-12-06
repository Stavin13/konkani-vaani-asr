"""
Evaluation metrics and reporting utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import List, Dict, Any


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary containing accuracy, precision, recall, and f1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def get_confusion_matrix(
    y_true: List[int],
    y_pred: List[int]
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str] = None
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of the classes
        
    Returns:
        Classification report as string
    """
    if target_names is None:
        target_names = ['negative', 'neutral', 'positive']
    
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
