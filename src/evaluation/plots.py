"""
src.evaluation.plots — Evaluation Visualization.

Generates:
    - ROC curve with AUC
    - Precision-Recall curve with AUC
    - Confusion matrix (heatmap)
    - Reliability / calibration plot
    - Training curves (loss + metrics over epochs)

All plots saved as PNG files to the configured output directory.
Results are from actual model runs — never fabricated.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def generate_all_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    training_history: Optional[Dict[str, List[float]]] = None,
    output_dir: str = "outputs/figures",
) -> List[str]:
    """Generate and save all evaluation plots.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    training_history : Dict[str, List[float]], optional
        Training curves data (epoch → metric values).
    output_dir : str
        Directory to save plot PNGs.

    Returns
    -------
    List[str]
        List of file paths to generated plots.
    """
    # TODO: Implement in Phase 6 (evaluation).
    raise NotImplementedError("Implement in Phase 6")
