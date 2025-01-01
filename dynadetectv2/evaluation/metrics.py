"""Evaluation metrics for DynaDetect v2."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, Any]:
    """Compute evaluation metrics.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(true_labels, predicted_labels, average='weighted'),
        'recall': recall_score(true_labels, predicted_labels, average='weighted'),
        'f1': f1_score(true_labels, predicted_labels, average='weighted')
    }
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Compute per-class metrics
    classes = np.unique(true_labels)
    per_class_metrics = {
        'precision_per_class': precision_score(true_labels, predicted_labels, average=None).tolist(),
        'recall_per_class': recall_score(true_labels, predicted_labels, average=None).tolist(),
        'f1_per_class': f1_score(true_labels, predicted_labels, average=None).tolist()
    }
    metrics.update(per_class_metrics)
    
    return metrics
