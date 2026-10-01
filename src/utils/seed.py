"""
src.utils.seed — Deterministic seeding for reproducibility.

Sets seeds for Python's ``random``, ``numpy``, and ``torch`` (CPU + CUDA).
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across all random number generators.

    Parameters
    ----------
    seed : int
        The seed value to use for all RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic algorithms where possible.
    # Note: some operations do not have deterministic implementations.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
