"""
src.models.hierarchical_pool — Hierarchical Attention Pooling.

Two-stage attention pooling to convert a variable-sized graph into a
single patient embedding, followed by missingness fusion.

Stage 1 — Type-Level Attention:
    EVENT NODES → EVENT TYPE REPRESENTATION
    For each patient p and event type τ:
        score_i = q_τ^T h_i
        α_i = softmax(score_i)  (over nodes of same patient AND same type)
        h_τ = Σ_i α_i · h_i

Stage 2 — Patient-Level Attention:
    EVENT TYPE REPRESENTATIONS → PATIENT EVENT EMBEDDING
    For each patient p:
        score_τ = q_p^T h_τ
        β_τ = softmax_masked(score_τ, mask=type_present_mask)
        h_p = Σ_τ β_τ · h_τ

Fusion:
    [h_p || z_miss] → Linear → ReLU → Dropout → fused_embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


class TypeLevelAttention(nn.Module):
    """Attention pooling over event nodes within the same type and patient.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node embeddings (default: 64).
    n_types : int
        Number of distinct event types (default: 6).
    """

    def __init__(self, hidden_dim: int = 64, n_types: int = 6) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_types = n_types

        # One learnable query vector per event type
        self.q_type = nn.Parameter(torch.empty(n_types, hidden_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_type)

    def forward(
        self, h: torch.Tensor, node_type: torch.Tensor, batch_index: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool node embeddings into type representations per patient.

        Parameters
        ----------
        h : torch.Tensor
            Node embeddings, shape ``[total_nodes, hidden_dim]``.
        node_type : torch.Tensor
            Integer type per node, shape ``[total_nodes]``.
        batch_index : torch.Tensor
            Patient ID per node, shape ``[total_nodes]``.
        batch_size : int
            Total number of patients in the batch.

        Returns
        -------
        type_embeddings : torch.Tensor
            Shape ``[batch_size, n_types, hidden_dim]``.
        type_present_mask : torch.Tensor
            Shape ``[batch_size, n_types]``.
        alpha : torch.Tensor
            Node-level attention weights, shape ``[total_nodes]``.
        """
        N = h.size(0)

        # Unique index for each (patient, type) pair
        # group_index values in [0, batch_size * n_types - 1]
        group_index = batch_index * self.n_types + node_type

        # Look up the correct query vector for each node
        q_i = self.q_type[node_type]  # [N, hidden_dim]

        # Compute dot-product score
        score_i = (q_i * h).sum(dim=-1)  # [N]

        # Softmax over the group
        num_groups = batch_size * self.n_types
        alpha_i = pyg_softmax(score_i, group_index, num_nodes=num_groups)  # [N]

        # Weighted sum per group
        out_flat = torch.zeros(num_groups, self.hidden_dim, device=h.device, dtype=h.dtype)
        msg = alpha_i.unsqueeze(-1) * h  # [N, hidden_dim]
        out_flat.scatter_add_(0, group_index.unsqueeze(-1).expand(-1, self.hidden_dim), msg)

        # Reshape to [batch_size, n_types, hidden_dim]
        type_embeddings = out_flat.view(batch_size, self.n_types, self.hidden_dim)

        # Compute mask indicating if a (patient, type) has any nodes
        counts = torch.zeros(num_groups, device=h.device, dtype=h.dtype)
        counts.scatter_add_(0, group_index, torch.ones(N, device=h.device, dtype=h.dtype))
        type_present_mask = (counts > 0).float().view(batch_size, self.n_types)

        return type_embeddings, type_present_mask, alpha_i


class PatientLevelAttention(nn.Module):
    """Attention pooling over type summaries to produce a patient embedding.

    Parameters
    ----------
    hidden_dim : int
        Dimension of type representations (default: 64).
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Global learned query vector for the patient
        self.q_patient = nn.Parameter(torch.empty(hidden_dim))
        self._init_weights()

    def _init_weights(self):
        # Initialise like a linear layer weight
        nn.init.uniform_(self.q_patient, -0.1, 0.1)

    def forward(
        self, type_embeddings: torch.Tensor, type_present_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool type summaries into a patient embedding.

        Parameters
        ----------
        type_embeddings : torch.Tensor
            Shape ``[batch_size, n_types, hidden_dim]``.
        type_present_mask : torch.Tensor
            Shape ``[batch_size, n_types]``.

        Returns
        -------
        patient_embedding : torch.Tensor
            Shape ``[batch_size, hidden_dim]``.
        beta : torch.Tensor
            Type-level attention weights, shape ``[batch_size, n_types]``.
        """
        # score_tau: [batch_size, n_types]
        score_tau = (type_embeddings * self.q_patient).sum(dim=-1)

        # Mask out absent types with -inf before softmax
        # If all types are absent (empty graph), softmax will yield uniform over types.
        # We handle this by multiplying beta by the mask afterwards.
        score_tau = score_tau.masked_fill(type_present_mask == 0, -1e9)
        
        beta = torch.softmax(score_tau, dim=-1)
        
        # Ensure beta is exactly 0 for absent types (in case of empty graphs or numerical issues)
        beta = beta * type_present_mask
        
        # For completely empty graphs, beta sum is 0, which is correct.
        
        # h_p: [batch_size, hidden_dim]
        h_p = (beta.unsqueeze(-1) * type_embeddings).sum(dim=1)
        
        return h_p, beta


class HierarchicalAttentionPool(nn.Module):
    """Hierarchical graph pooling and missingness fusion.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node embeddings and intermediate embeddings.
    n_types : int
        Number of distinct event types.
    missingness_dim : int
        Dimension of the missingness summary z_miss.
    fused_dim : int
        Output dimension after fusing graph and missingness representations.
    dropout : float
        Dropout probability in the fusion layer.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_types: int = 6,
        missingness_dim: int = 64,
        fused_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.type_attn = TypeLevelAttention(hidden_dim, n_types)
        self.patient_attn = PatientLevelAttention(hidden_dim)
        
        # Fusion projection: [h_p || z_miss] -> fused_dim
        fusion_in_dim = hidden_dim + missingness_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        h: torch.Tensor,
        node_type: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
        z_miss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool node embeddings and fuse with missingness representation.

        Parameters
        ----------
        h : torch.Tensor
            Node embeddings, [total_nodes, hidden_dim].
        node_type : torch.Tensor
            Integer type per node, [total_nodes].
        batch_index : torch.Tensor
            Patient ID per node, [total_nodes].
        batch_size : int
            Total number of patients in the batch.
        z_miss : torch.Tensor
            Missingness summary, [batch_size, missingness_dim].

        Returns
        -------
        fused_embedding : torch.Tensor
            Fused patient representation, [batch_size, fused_dim].
        patient_event_embedding : torch.Tensor
            Pooled event representation h_p, [batch_size, hidden_dim].
        type_embeddings : torch.Tensor
            Type representations h_tau, [batch_size, n_types, hidden_dim].
        type_present_mask : torch.Tensor
            Mask of present types, [batch_size, n_types].
        type_attention_weights : torch.Tensor
            Patient-level attention beta, [batch_size, n_types].
        node_type_attention : torch.Tensor
            Type-level attention alpha, [total_nodes].
        """
        # Stage 1: Nodes -> Types
        type_embeddings, type_present_mask, alpha = self.type_attn(
            h, node_type, batch_index, batch_size
        )
        
        # Stage 2: Types -> Patient
        h_p, beta = self.patient_attn(type_embeddings, type_present_mask)
        
        # Fusion
        cat = torch.cat([h_p, z_miss], dim=-1)
        fused = self.fusion(cat)
        
        return fused, h_p, type_embeddings, type_present_mask, beta, alpha
