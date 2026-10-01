"""
src.explainability.graph_explain — Graph Substructure Explanation (Level 3).

Uses GNNExplainer or PGExplainer to identify influential nodes and
edges in the patient's temporal event graph.

These are ASSOCIATIONAL substructure explanations, not causal.
"""

from typing import Dict, Tuple

import torch
from torch_geometric.data import Data


class GraphSubstructureExplainer:
    """Identify influential graph substructures for a prediction.

    Parameters
    ----------
    model : torch.nn.Module
        Trained HITGNN model.
    epochs : int
        Number of GNNExplainer optimization epochs (default: 200).
    lr : float
        Learning rate for GNNExplainer (default: 0.01).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> None:
        # TODO: Implement in Phase 7 (explainability).
        raise NotImplementedError("Implement in Phase 7")

    def explain(
        self,
        patient_graph: Data,
        missingness_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate substructure explanation for a patient.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``"node_mask"``: soft mask over nodes, shape ``[num_nodes]``
            ``"edge_mask"``: soft mask over edges, shape ``[num_edges]``
        """
        raise NotImplementedError("Implement in Phase 7")
