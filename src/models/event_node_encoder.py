"""
src.models.event_node_encoder — Initial Node Feature Encoder.

Converts raw per-node attributes from the temporal event graph into a
learned dense embedding h_v^(0) ∈ ℝ^hidden_dim, which is the input to TE-GAT.

Input attributes (from PyG Data):
    data.node_type       [N]  — integer node-type ID  (0–5)
    data.concept_id      [N]  — integer canonical concept ID
    data.numeric_value   [N]  — train-normalised value (or 0 for non-numeric)
    data.has_value       [N]  — 1.0 if node carries a numeric measurement, else 0.0
    data.relative_time   [N]  — hours relative to t0 (≤ 0)

Architecture per node v:
    e_type    = TypeEmbedding(node_type_v)             # [type_dim]
    e_concept = ConceptEmbedding(concept_id_v)         # [concept_dim]
    e_numeric = Linear(2 → numeric_dim) → ReLU         # input: [numeric_value, has_value]
    φ_node(t) = TimeEncoding(relative_time_v)          # [time_dim]

    cat = [e_type || e_concept || e_numeric || φ_node(t)]   # [sum of dims]
    h_v^(0) = Linear(sum_dims → hidden_dim) → ReLU          # [hidden_dim]

Design notes:
    - node_type and concept_id remain INTEGER tensors; embeddings handle
      lookup — no manual one-hot construction.
    - Mortality label, latent severity, and split identity are NOT included.
    - numeric_value=0, has_value=0 for MEDICATION/PROCEDURE/DIAGNOSIS/
      CLINICAL_CONCEPT nodes (structural zero, not a clinical zero).
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.time_encoding import TimeEncoding


# Node type registry (must match graph_builder.NODE_TYPE_MAP)
NUM_NODE_TYPES: int = 6   # LAB, VITAL, MEDICATION, PROCEDURE, DIAGNOSIS, CLINICAL_CONCEPT


class EventNodeEncoder(nn.Module):
    """Encode raw node attributes into a dense initial embedding h_v^(0).

    Parameters
    ----------
    num_concepts : int
        Total size of the canonical concept vocabulary.
    hidden_dim : int
        Output embedding dimension per node (default: 64).
    type_embedding_dim : int
        Dimension of the node-type embedding (default: 16).
    concept_embedding_dim : int
        Dimension of the concept embedding (default: 32).
    numeric_hidden_dim : int
        Hidden dimension for the numeric value MLP (default: 16).
    time_dim : int
        Dimension of the node time encoding (default: 16).
    """

    def __init__(
        self,
        num_concepts: int,
        hidden_dim: int = 64,
        type_embedding_dim: int = 16,
        concept_embedding_dim: int = 32,
        numeric_hidden_dim: int = 16,
        time_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Categorical embeddings ---
        self.type_emb = nn.Embedding(NUM_NODE_TYPES, type_embedding_dim)
        self.concept_emb = nn.Embedding(num_concepts, concept_embedding_dim)

        # --- Numeric branch ---
        # Input: [numeric_value (normalised), has_value indicator]
        self.numeric_proj = nn.Sequential(
            nn.Linear(2, numeric_hidden_dim),
            nn.ReLU(),
        )

        # --- Node time encoding ---
        self.node_time_enc = TimeEncoding(output_dim=time_dim)

        # --- Fusion projection ---
        fusion_dim = type_embedding_dim + concept_embedding_dim + numeric_hidden_dim + time_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Encode all nodes in a (possibly batched) patient graph.

        Parameters
        ----------
        data : Data
            PyG Data (or Batch) with node attributes.

        Returns
        -------
        torch.Tensor
            Initial node embeddings, shape ``[N, hidden_dim]``.
        """
        # Categorical components
        e_type = self.type_emb(data.node_type)             # [N, type_dim]
        e_concept = self.concept_emb(data.concept_id)      # [N, concept_dim]

        # Numeric component — safe for non-numeric nodes (has_value=0)
        numeric_in = torch.stack(
            [data.numeric_value, data.has_value], dim=-1
        )                                                   # [N, 2]
        e_numeric = self.numeric_proj(numeric_in)          # [N, numeric_dim]

        # Node time component
        e_time = self.node_time_enc(data.relative_time)   # [N, time_dim]

        # Concatenate and project
        cat = torch.cat([e_type, e_concept, e_numeric, e_time], dim=-1)
        return self.fusion(cat)                             # [N, hidden_dim]
