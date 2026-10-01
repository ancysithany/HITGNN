"""
src.graph.population_graph — Inductive Inter-Patient Population Graph.

Constructs a k-NN graph connecting patients with clinically similar
TRAINING patients.

Similarity metric:
    s(p, q) = λ₁ · cosine(h_p, h_q) + λ₂ · similarity(δ_p, δ_q)

where:
    h_p, h_q = learned patient embeddings
    δ_p, δ_q = patient metadata vectors (age, SOFA, ICU type, comorbidities)
    λ₁, λ₂ = configurable weights (default 0.7, 0.3)

CRITICAL INDUCTIVE PROTOCOL:
    ┌─────────────┬──────────────────┬───────────────────────┐
    │ Phase       │ Reference Bank   │ Allowed Edges         │
    ├─────────────┼──────────────────┼───────────────────────┤
    │ Training    │ Train patients   │ Train ↔ Train         │
    │ Validation  │ Train (frozen)   │ Val → Train ONLY      │
    │ Test        │ Train (frozen)   │ Test → Train ONLY     │
    └─────────────┴──────────────────┴───────────────────────┘

    NEVER:
    - Validation → Validation edges
    - Test → Test edges
    - Outcome labels used in similarity computation
    - Test outcome labels accessed during neighbor retrieval
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data


class PopulationGraphBuilder:
    """Build the inductive inter-patient similarity graph.

    Parameters
    ----------
    config : dict
        The ``population_graph`` section of the configuration.
    """

    def __init__(self, config: dict) -> None:
        # TODO: Implement in Phase 4 (population graph).
        raise NotImplementedError("Implement in Phase 4")

    def build_training_graph(
        self,
        train_embeddings: torch.Tensor,
        train_metadata: List[Dict],
    ) -> Data:
        """Build population graph over training patients.

        Parameters
        ----------
        train_embeddings : torch.Tensor
            Patient embeddings, shape ``[n_train, embed_dim]``.
        train_metadata : List[Dict]
            List of metadata dicts (age, sofa, icu_type, comorbidities).

        Returns
        -------
        torch_geometric.data.Data
            Population graph with k-NN edges among training patients.
        """
        raise NotImplementedError("Implement in Phase 4")

    def build_inductive_graph(
        self,
        query_embeddings: torch.Tensor,
        query_metadata: List[Dict],
        reference_embeddings: torch.Tensor,
        reference_metadata: List[Dict],
    ) -> Data:
        """Build inductive graph: query patients → training reference bank.

        Query patients (val/test) connect ONLY to training patients.
        No query-to-query edges are created.

        Parameters
        ----------
        query_embeddings : torch.Tensor
            Embeddings of val/test patients, shape ``[n_query, embed_dim]``.
        query_metadata : List[Dict]
            Metadata for query patients.
        reference_embeddings : torch.Tensor
            FROZEN training patient embeddings, shape ``[n_ref, embed_dim]``.
        reference_metadata : List[Dict]
            Metadata for training (reference) patients.

        Returns
        -------
        torch_geometric.data.Data
            Population graph where query nodes connect only to reference
            (training) nodes.
        """
        raise NotImplementedError("Implement in Phase 4")

    def _compute_similarity(
        self,
        h_p: torch.Tensor,
        h_q: torch.Tensor,
        meta_p: Dict,
        meta_q: Dict,
    ) -> float:
        """Compute pairwise patient similarity.

        s(p, q) = λ₁ · cosine(h_p, h_q) + λ₂ · sim(δ_p, δ_q)

        Outcome labels are NEVER used in this computation.
        """
        raise NotImplementedError("Implement in Phase 4")
