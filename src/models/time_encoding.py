"""
src.models.time_encoding — Relative / Absolute Time Encoding φ(t).

Encodes continuous time values (hours) into a fixed-dimensional dense vector
using learnable sinusoidal basis functions (inspired by Time2Vec).

Convention:
    φ(t) = W_proj · [sin(ω₁·t + b₁), cos(ω₁·t + b₁), ...,
                      sin(ω_F·t + b_F), cos(ω_F·t + b_F)]  ∈ ℝ^output_dim

where ω and b are learnable parameters and W_proj is a linear projection.

Properties:
    - Supports any input shape (broadcast-safe via view/expand)
    - Handles zero time gaps and large gaps without NaN/Inf
    - Differentiable: gradients flow through ω, b, W_proj
    - Device-safe: follows module device automatically
    - Distinct inputs produce distinct outputs (injective in practice)

Shared by:
    A. EdgeTimeEncoding  — encodes edge_delta_t for TE-GAT attention logits
    B. NodeTimeEncoding  — encodes relative_time_to_t0 for initial node features
"""

import math

import torch
import torch.nn as nn


class TimeEncoding(nn.Module):
    """Learnable sinusoidal time encoding for edges or nodes.

    Parameters
    ----------
    output_dim : int
        Dimension of the output time encoding vector.
    n_frequencies : int
        Number of sinusoidal frequency pairs (sin/cos).
        The raw sinusoidal output has shape [*, 2*n_frequencies], which
        is then projected to [*, output_dim].
    """

    def __init__(
        self,
        output_dim: int = 16,
        n_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.n_frequencies = n_frequencies

        # Learnable frequencies and phases, one per frequency
        self.omega = nn.Parameter(torch.empty(n_frequencies))
        self.phase = nn.Parameter(torch.empty(n_frequencies))

        # Linear projection: 2*n_frequencies -> output_dim
        self.proj = nn.Linear(2 * n_frequencies, output_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise frequencies to geometric spacing (like positional encoding)."""
        # Frequencies in [1/(max_t), 1] on log scale — max_t ~ 200 hours
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0 / 200.0),
                math.log(1.0),
                self.n_frequencies,
            )
        )
        self.omega.data.copy_(freqs)
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode a time tensor.

        Parameters
        ----------
        t : torch.Tensor
            Time values in hours, shape ``[*]``.

        Returns
        -------
        torch.Tensor
            Time encodings, shape ``[*, output_dim]``.
        """
        # t: [*] -> [*, 1] for broadcasting with [n_freq]
        t_exp = t.unsqueeze(-1)                           # [*, 1]
        angles = t_exp * self.omega + self.phase          # [*, n_freq]
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [*, 2*n_freq]
        return self.proj(sincos)                          # [*, output_dim]
