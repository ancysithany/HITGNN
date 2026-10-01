"""
src.evaluation.metrics — Evaluation Metrics for Mortality Prediction.

Computes:
    - AUROC
    - AUPRC
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - F1 Score
    - Brier Score

All metrics are computed from real model predictions on synthetic data.
Results are NEVER fabricated, hardcoded, or claimed to reproduce
results from the research study.
"""

from typing import Dict

import numpy as np


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape ``[n_samples]``.
    y_prob : np.ndarray
        Predicted probabilities (after sigmoid), shape ``[n_samples]``.
    threshold : float
        Classification threshold for binary metrics (default: 0.5).

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric name to computed value.
    """
    # TODO: Implement in Phase 6 (evaluation).
    raise NotImplementedError("Implement in Phase 6")
