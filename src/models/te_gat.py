"""
src.models.te_gat — Temporal Edge-Aware Graph Attention Network (TE-GAT).

Implements a multi-head graph attention mechanism where relative temporal
information participates DIRECTLY in the attention logit, not just in the
message.

Attention formulation per edge u→v (per head h):
    e_uv^h = LeakyReLU(
        a_h^T · [ W_h · h_v  ||  W_h · h_u  ||  φ_edge(Δt_uv)  ||  e_etype ]
    )
    α_uv^h = softmax_{u ∈ N(v)}(e_uv^h)      [graph-local, via PyG softmax]
    m_v^h  = Σ_u α_uv^h · W_msg_h · h_u
    h_v'   = ELU( Linear_h([m_v^1 || … || m_v^H]) ) + residual(h_v)

DESIGN DECISIONS:
    1. Time is concatenated into the attention logit vector (not added post-hoc).
    2. Edge type is embedded and also concatenated into the attention logit.
    3. Residual connection: if in_dim != out_dim, a learned Linear projection
       is applied to h_v before adding.
    4. No self-loops added: the residual path already provides self-information
       without mutating stored graph topology.
    5. Attention normalisation uses torch_geometric.utils.softmax with the
       destination index to remain graph-local (no cross-patient contamination
       in batched operation).
    6. Multi-head: intermediate layers concatenate heads and project back to
       hidden_dim; the final layer also projects to hidden_dim.
    7. Attention records can be extracted for explainability without detach.

Layer stacking (TEGATEncoder):
    h0 → TEGATLayer(concat=True)  → project to hidden_dim → h1
    h1 → TEGATLayer(concat=True)  → project to hidden_dim → h2
    h2 → TEGATLayer(concat=True)  → project to hidden_dim → h3 (output)

NUM_EDGE_TYPES = 4  (TEMPORAL_ADJACENCY, SAME_CONCEPT, SEMANTIC, CROSS_MODALITY)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from src.models.time_encoding import TimeEncoding


NUM_EDGE_TYPES = 4


class TemporalEdgeAttentionLayer(MessagePassing):
    """Single TE-GAT attention layer.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Per-head output dimension. Full output = out_dim * n_heads.
    time_dim : int
        Dimension of edge time encoding φ(Δt).
    edge_type_dim : int
        Dimension of edge-type embedding.
    n_heads : int
        Number of attention heads.
    leaky_relu_slope : float
        Negative slope for LeakyReLU in attention logit.
    dropout : float
        Dropout rate applied to attention coefficients during training.
    residual : bool
        Apply residual connection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_dim: int = 16,
        edge_type_dim: int = 8,
        n_heads: int = 4,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.3,
        residual: bool = True,
    ) -> None:
        # aggr="add" — we manually scale messages by alpha inside message(),
        # so summing is equivalent to the weighted aggregation.
        super().__init__(aggr="add", node_dim=0)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.time_dim = time_dim
        self.edge_type_dim = edge_type_dim
        self.slope = leaky_relu_slope
        self.residual = residual

        # Per-head: linear transforms for node features
        # W_h maps h → out_dim for each head
        self.W_node = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        # W_msg_h — separate message transform per head
        self.W_msg = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        # Edge time encoding (shared across heads; then the projection a uses time_dim)
        self.time_enc = TimeEncoding(output_dim=time_dim)

        # Edge type embedding (one per type, shared across heads)
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, edge_type_dim)

        # Attention vector a_h: per head, projects
        #   [out_dim (v) || out_dim (u) || time_dim || edge_type_dim]  → scalar
        attn_in = 2 * out_dim + time_dim + edge_type_dim
        self.attn_vec = nn.Parameter(torch.empty(n_heads, attn_in))

        # Attention dropout
        self.attn_drop = nn.Dropout(dropout)

        # Output projection: [out_dim * n_heads] → (out_dim * n_heads) kept, no compression here
        # Compression (back to hidden_dim) happens in TEGATEncoder.

        # Residual: project input if dims differ
        full_out = out_dim * n_heads
        if residual:
            if in_dim != full_out:
                self.res_proj = nn.Linear(in_dim, full_out, bias=False)
            else:
                self.res_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.W_msg.weight)
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0)).squeeze(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_delta_t: torch.Tensor,
        edge_type: torch.Tensor,
        return_attention: bool = False,
        static_gnn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, in_dim].
        edge_index : torch.Tensor
            COO edge list [2, E].
        edge_delta_t : torch.Tensor
            Edge time gaps in hours [E].
        edge_type : torch.Tensor
            Edge type integer IDs [E].
        return_attention : bool
            If True, return attention weights alongside embeddings.
        static_gnn : bool
            If True, zero out time encoding (ablation).

        Returns
        -------
        h : torch.Tensor
            Updated node embeddings [N, out_dim * n_heads].
        attn_info : dict or None
            Contains 'alpha' [E, n_heads], 'edge_index' [2, E] when requested.
        """
        N = x.size(0)

        # Linear transform for all heads at once
        Wh = self.W_node(x).view(N, self.n_heads, self.out_dim)   # [N, H, D]

        # Encode edge time and type
        if static_gnn:
            phi_t = torch.zeros((edge_delta_t.size(0), self.time_dim), device=x.device, dtype=x.dtype)
        else:
            phi_t = self.time_enc(edge_delta_t)                         # [E, time_dim]
        e_etype = self.edge_type_emb(edge_type)                     # [E, edge_type_dim]

        # Propagate — propagate packs data via message/aggregate/update
        # We pass Wh (transformed node features) to propagate.
        # MessagePassing will call self.message() with source/dest Wh values.
        out, alpha = self._propagate_with_attn(
            x, Wh, edge_index, phi_t, e_etype
        )                                                            # [N, H*D]

        # Residual
        if self.residual:
            out = F.elu(out) + self.res_proj(x)
        else:
            out = F.elu(out)

        attn_info = None
        if return_attention:
            attn_info = {"alpha": alpha, "edge_index": edge_index}

        return out, attn_info

    def _propagate_with_attn(
        self,
        x: torch.Tensor,
        Wh: torch.Tensor,
        edge_index: torch.Tensor,
        phi_t: torch.Tensor,
        e_etype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manual propagation to expose attention weights."""
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)
        E = src.size(0)

        # W_h h_u  and  W_h h_v   (source and destination)
        Wh_src = Wh[src]   # [E, H, D]
        Wh_dst = Wh[dst]   # [E, H, D]

        # Expand time and type encodings to per-head context
        # phi_t: [E, time_dim], e_etype: [E, edge_type_dim]
        # Replicate along head dim
        phi_t_exp = phi_t.unsqueeze(1).expand(-1, self.n_heads, -1)      # [E, H, time_dim]
        e_etype_exp = e_etype.unsqueeze(1).expand(-1, self.n_heads, -1)  # [E, H, etype_dim]

        # Concatenate attention inputs per head: [Wh_v || Wh_u || phi_t || e_etype]
        attn_in = torch.cat([Wh_dst, Wh_src, phi_t_exp, e_etype_exp], dim=-1)  # [E, H, attn_in_dim]

        # Compute attention logit for each head: a_h^T · attn_in_h
        # attn_vec: [H, attn_in_dim]
        e_uv = (attn_in * self.attn_vec.unsqueeze(0)).sum(dim=-1)  # [E, H]
        e_uv = F.leaky_relu(e_uv, negative_slope=self.slope)       # [E, H]

        # Neighbor-softmax per head: normalise over incoming edges for each dst node
        alpha = torch.stack([
            softmax(e_uv[:, h], dst, num_nodes=N)
            for h in range(self.n_heads)
        ], dim=1)                                                    # [E, H]

        alpha = self.attn_drop(alpha)

        # Message: alpha_uv^h * W_msg_h h_u
        Wmsg_src = self.W_msg(x)[src].view(E, self.n_heads, self.out_dim)  # [E, H, D]
        msgs = alpha.unsqueeze(-1) * Wmsg_src                               # [E, H, D]

        # Aggregate (sum) per destination node
        msgs_flat = msgs.view(E, self.n_heads * self.out_dim)               # [E, H*D]

        # scatter_add to aggregate per dest node
        out = torch.zeros(N, self.n_heads * self.out_dim, device=x.device, dtype=x.dtype)
        dst_exp = dst.unsqueeze(1).expand(-1, self.n_heads * self.out_dim)
        out.scatter_add_(0, dst_exp, msgs_flat)                             # [N, H*D]

        return out, alpha


class TEGATEncoder(nn.Module):
    """Stack of TE-GAT layers producing per-node embeddings.

    Pipeline::

        h0 = EventNodeEncoder(data)           [N, in_dim]
        h1 = TEGATLayer(h0)  → compress       [N, hidden_dim]
        h2 = TEGATLayer(h1)  → compress       [N, hidden_dim]
        h3 = TEGATLayer(h2)  → compress       [N, hidden_dim]   ← output

    Parameters
    ----------
    in_dim : int
        Input dimension from EventNodeEncoder (= hidden_dim there).
    hidden_dim : int
        Per-node hidden dimension to maintain throughout (default: 64).
    n_heads : int
        Attention heads per layer (default: 4).
    n_layers : int
        Number of stacked TE-GAT layers (default: 3).
    time_dim : int
        Edge time encoding output dimension (default: 16).
    edge_type_dim : int
        Edge-type embedding dimension (default: 8).
    leaky_relu_slope : float
        LeakyReLU slope for attention logits (default: 0.2).
    dropout : float
        Attention dropout (default: 0.3).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        time_dim: int = 16,
        edge_type_dim: int = 8,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.time_dim = time_dim
        self.edge_type_dim = edge_type_dim

        # Build layers; each TE-GAT produces out_dim*n_heads, then we compress back.
        # We set out_dim = hidden_dim // n_heads so that out_dim*n_heads == hidden_dim.
        per_head_dim = hidden_dim // n_heads

        layers = []
        compressors = []
        layer_in = in_dim
        for i in range(n_layers):
            layers.append(
                TemporalEdgeAttentionLayer(
                    in_dim=layer_in,
                    out_dim=per_head_dim,
                    time_dim=time_dim,
                    edge_type_dim=edge_type_dim,
                    n_heads=n_heads,
                    leaky_relu_slope=leaky_relu_slope,
                    dropout=dropout,
                    residual=True,
                )
            )
            # per_head_dim * n_heads = hidden_dim → already correct; project for clean interface
            compressors.append(nn.Linear(per_head_dim * n_heads, hidden_dim, bias=False))
            layer_in = hidden_dim  # subsequent layers receive hidden_dim

        self.layers = nn.ModuleList(layers)
        self.compress = nn.ModuleList(compressors)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_delta_t: torch.Tensor,
        edge_type: torch.Tensor,
        return_attention: bool = False,
        static_gnn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """Forward through all TE-GAT layers.

        Parameters
        ----------
        x : torch.Tensor
            Initial node embeddings from EventNodeEncoder, [N, in_dim].
        edge_index : torch.Tensor
            COO edge list [2, E].
        edge_delta_t : torch.Tensor
            Edge delta-t values in hours [E].
        edge_type : torch.Tensor
            Edge type integer IDs [E].
        return_attention : bool
            If True, collect and return attention records from every layer.
        static_gnn : bool
            If True, zero out time encoding (ablation).

        Returns
        -------
        h : torch.Tensor
            Final node embeddings [N, hidden_dim].
        attn_records : list of dict or None
            One dict per layer, each containing 'alpha', 'edge_index', 'layer'.
        """
        h = x
        attn_records = [] if return_attention else None

        for i, (layer, comp) in enumerate(zip(self.layers, self.compress)):
            h_new, attn_info = layer(
                h, edge_index, edge_delta_t, edge_type,
                return_attention=return_attention,
                static_gnn=static_gnn,
            )
            h = comp(h_new)        # [N, hidden_dim]
            h = self.drop(h)

            if return_attention and attn_info is not None:
                attn_info["layer"] = i
                attn_info["n_heads"] = self.n_heads
                attn_records.append(attn_info)

        return h, attn_records
