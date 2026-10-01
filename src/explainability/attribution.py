"""
src.explainability.attribution — Event Attribution (Levels 1 & 2).

Level 1: Event Attribution
    Uses Integrated Gradients (or similar) to compute per-node/feature
    importance scores for a given patient prediction.

Level 2: Temporal Saliency
    Arranges attributed events chronologically to visualize which
    time points contributed most to the prediction.

These are ASSOCIATIONAL attributions, not causal explanations.
"""

from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data


class EventAttribution:
    """Compute event-level attributions for a patient's mortality prediction.

    Parameters
    ----------
    model : torch.nn.Module
        Trained HITGNN model.
    n_steps : int
        Number of interpolation steps for Integrated Gradients (default: 50).
    """

    def __init__(self, model: torch.nn.Module, n_steps: int = 50) -> None:
        # TODO: Implement in Phase 7 (explainability).
        raise NotImplementedError("Implement in Phase 7")

    def compute_attributions(
        self,
        patient_graph: Data,
        missingness_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-node attribution scores.

        Parameters
        ----------
        patient_graph : Data
            Patient event graph.
        missingness_input : torch.Tensor
            Missingness encoder input.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``"node_attributions"``: per-node importance, shape ``[num_nodes]``
            ``"feature_attributions"``: per-feature importance, shape ``[num_nodes, feat_dim]``
        """
        raise NotImplementedError("Implement in Phase 7")

    def temporal_saliency(
        self,
        patient_graph: Data,
        node_attributions: torch.Tensor,
    ) -> List[Tuple[float, str, float]]:
        """Arrange attributed events chronologically (Level 2).

        Returns
        -------
        List[Tuple[float, str, float]]
            List of ``(timestamp, concept_name, attribution_score)``
            sorted by timestamp.
        """
        raise NotImplementedError("Implement in Phase 7")
