"""
src.models.missingness_encoder — Missingness-Aware Biomarker Encoder.

Encodes a fixed-vocabulary biomarker panel with explicit missingness
representation.  Each biomarker x temporal-bin slot is represented as:

    input_bk = [x_bk * m_bk, m_bk, log(1 + delta_t_bk), n_bk]

where:
    x_bk    = observed numerical value (normalized)
    m_bk    = observation mask in {0, 1}
    delta_t = time since most recent same-biomarker measurement (hours)
    n_bk    = never-observed indicator in {0, 1}

When a measurement is absent:
    m_bk = 0  =>  x_bk * m_bk = 0
    No population mean substitution occurs.

Architecture:
    Shared 2-layer MLP:
        Linear(4, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)

    The SAME MLP parameters are shared across all biomarkers and all
    temporal bins.  This is achieved by reshaping the input to collapse
    the K and B dimensions before applying the MLP.

    Optional output activation is configurable but defaults to None.

Terminology:
    This is "missingness-aware modeling."
    It does NOT claim to formally solve MNAR missingness.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MissingnessEncoder(nn.Module):
    """Missingness-aware biomarker encoder.

    Parameters
    ----------
    input_dim : int
        Dimension of per-slot input vector (default: 4).
    hidden_dim : int
        Hidden layer dimension (default: 64).
    output_dim : int
        Output embedding dimension (default: 64).
    n_biomarkers : int
        Number of biomarkers in the fixed panel (default: 7).
    n_bins : int
        Number of temporal bins (default: 6).
    output_activation : str or None
        Optional activation on the output layer.
        ``None`` (default), ``"relu"``, or ``"tanh"``.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_biomarkers: int = 7,
        n_bins: int = 6,
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_biomarkers = n_biomarkers
        self.n_bins = n_bins

        # Shared 2-layer MLP — same parameters for all biomarkers and bins
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Optional output activation
        if output_activation == "relu":
            self.out_act = nn.ReLU()
        elif output_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

    def forward(
        self, missingness_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the missingness representation for a batch of patients.

        Parameters
        ----------
        missingness_input : torch.Tensor
            Shape ``[batch_size, K, B, 4]``.
            Channels: [x*mask, mask, log(1+delta_t), never_observed].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - ``slot_embeddings``: Shape ``[batch_size, K, B, D]``.
              Per-slot embeddings (retained for attention/explainability).
            - ``z_miss``: Shape ``[batch_size, D]``.
              Patient-level summary via mean pooling over K and B.
        """
        batch_size = missingness_input.size(0)
        K = missingness_input.size(1)
        B = missingness_input.size(2)

        # Reshape to [batch * K * B, 4] — shared MLP across all slots
        flat = missingness_input.reshape(-1, self.input_dim)

        # Apply shared MLP
        encoded = self.mlp(flat)            # [batch * K * B, D]
        encoded = self.out_act(encoded)

        # Reshape back to [batch, K, B, D]
        slot_embeddings = encoded.reshape(batch_size, K, B, self.output_dim)

        # Patient-level summary: mean over K and B dimensions
        # Every slot intentionally exists (even when unobserved), so
        # plain mean is appropriate — the MLP learns to handle the
        # mask/never_observed indicators in the input channels.
        z_miss = slot_embeddings.mean(dim=(1, 2))  # [batch, D]

        return slot_embeddings, z_miss


class ValueOnlyEncoder(nn.Module):
    """Value-only mean-imputation biomarker encoder for ablation.
    
    Receives only the value channel (imputed with 0.0 when missing,
    which corresponds to the train-set mean).
    Does NOT receive mask, delta_t, or never_observed.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_biomarkers: int = 7,
        n_bins: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, missingness_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # missingness_input: [batch, K, B, 4]
        # We only take the first channel (value).
        # Since unobserved slots have value=0.0 (the z-score normalized mean),
        # this is equivalent to train-set mean imputation.
        values_only = missingness_input[..., 0:1] # [batch, K, B, 1]
        
        batch_size = values_only.size(0)
        K = values_only.size(1)
        B = values_only.size(2)
        
        flat = values_only.reshape(-1, self.input_dim)
        encoded = self.mlp(flat)
        slot_embeddings = encoded.reshape(batch_size, K, B, self.output_dim)
        z_miss = slot_embeddings.mean(dim=(1, 2))
        
        return slot_embeddings, z_miss
