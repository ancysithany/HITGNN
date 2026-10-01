"""
src.training.metrics — Evaluation metrics for Phase 7.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate all Phase 7 evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities (output of sigmoid), range [0, 1].
    threshold : float
        Threshold for binary classification.
        
    Returns
    -------
    dict
        Dictionary containing auprc, auroc, accuracy, precision,
        sensitivity, specificity, f1, and brier_score.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    
    # AUROC / AUPRC
    try:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auroc'] = 0.5  # Fallback if only one class present
        
    try:
        metrics['auprc'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['auprc'] = 0.0
        
    # Brier Score
    try:
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
    except ValueError:
        metrics['brier_score'] = 1.0
        
    # Threshold-dependent metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0) # Same as recall
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        # Happens if only one class in y_true and y_pred
        metrics['specificity'] = 0.0
        
    return metrics


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, strategy: str = "validation_f1") -> float:
    """Find the best threshold based on a strategy.
    
    Parameters
    ----------
    y_true : np.ndarray
        Validation binary labels.
    y_prob : np.ndarray
        Validation predicted probabilities.
    strategy : str
        Strategy name, e.g. "validation_f1".
        
    Returns
    -------
    float
        Selected threshold.
    """
    if strategy != "validation_f1":
        # Default to 0.5 if not maximizing F1
        return 0.5
        
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1
    best_thresh = 0.5
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    return float(best_thresh)
