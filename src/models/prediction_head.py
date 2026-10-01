"""
src.models.prediction_head — Final Mortality Prediction Head.

Architecture:
    Linear(input_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → 1)

DESIGN DECISION:
    During TRAINING, this module outputs RAW LOGITS (no sigmoid).
    Use ``BCEWithLogitsLoss(logit, label)`` for numerically stable training.

    For INFERENCE/EVALUATION only:
        probability = torch.sigmoid(logit)

    The model's ``forward()`` never applies sigmoid internally when
    used with the training loss.

Output interpretation:
    The output is a "predicted mortality probability" (after sigmoid).
    Do NOT call this "confidence" — probability and model confidence
    are not interchangeable.
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """Binary mortality prediction head outputting raw logits.

    Parameters
    ----------
    input_dim : int
        Dimension of the patient representation (default: 128).
    hidden_dim : int
        Hidden layer dimension (default: 64).
    dropout : float
        Dropout rate (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Produce raw logit for mortality prediction.

        Parameters
        ----------
        h : torch.Tensor
            Patient representation, shape ``[batch_size, input_dim]``
            or ``[input_dim]`` for a single patient.

        Returns
        -------
        torch.Tensor
            Raw logit(s), shape ``[batch_size]`` or ``[]``.
            Apply ``torch.sigmoid()`` at inference only.
        """
        # Return shape [N] instead of [N, 1] as preferred by the spec
        return self.mlp(h).squeeze(-1)
