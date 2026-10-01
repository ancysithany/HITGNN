"""
src.preprocessing.normalizer — Feature normalization utilities.

Provides z-score and min-max normalization for continuous clinical
values.  Statistics are computed on the TRAINING split only and
applied consistently to validation/test splits to prevent leakage.

Only OBSERVED measurements contribute to mean/std.
Missing values (mask=0 slots) are never included in statistics.
"""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np


class FeatureNormalizer:
    """Fit-transform normalizer for continuous clinical values.

    Computes statistics (mean, std) from training data
    and applies the same transformation to all splits.

    Parameters
    ----------
    method : str
        Normalization method: ``"zscore"`` or ``"minmax"``.
    """

    def __init__(self, method: str = "zscore") -> None:
        self.method = method
        self._stats: Dict[str, Tuple[float, float]] = {}
        self._fitted: bool = False

    def fit(self, concept_name: str, values: np.ndarray) -> None:
        """Compute normalization statistics from training values.

        Parameters
        ----------
        concept_name : str
            Clinical concept name (e.g., ``"Lactate"``).
        values : np.ndarray
            Array of observed training values (NaNs excluded beforehand).
            Must contain only OBSERVED measurements — missing values must
            NOT be included.
        """
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            # No observations — use identity transform
            self._stats[concept_name] = (0.0, 1.0)
            return

        if self.method == "zscore":
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std < 1e-8:
                std = 1.0  # Avoid division by zero
            self._stats[concept_name] = (mean, std)
        elif self.method == "minmax":
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            rng = vmax - vmin
            if rng < 1e-8:
                rng = 1.0
            self._stats[concept_name] = (vmin, rng)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self._fitted = True

    def transform(self, concept_name: str, value: float) -> float:
        """Normalize a single value using fitted statistics.

        Parameters
        ----------
        concept_name : str
            Clinical concept name.
        value : float
            Raw measurement value.

        Returns
        -------
        float
            Normalized value.
        """
        if concept_name not in self._stats:
            return value  # Passthrough if not fitted for this concept

        if self.method == "zscore":
            mean, std = self._stats[concept_name]
            return (value - mean) / std
        elif self.method == "minmax":
            vmin, rng = self._stats[concept_name]
            return (value - vmin) / rng
        return value

    def transform_array(self, concept_name: str, values: np.ndarray) -> np.ndarray:
        """Normalize an array of values."""
        if concept_name not in self._stats:
            return values
        if self.method == "zscore":
            mean, std = self._stats[concept_name]
            return (values - mean) / std
        elif self.method == "minmax":
            vmin, rng = self._stats[concept_name]
            return (values - vmin) / rng
        return values

    def fit_transform(
        self, concept_name: str, values: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(concept_name, values)
        return self.transform_array(concept_name, values)

    @property
    def is_fitted(self) -> bool:
        """Whether any statistics have been computed."""
        return self._fitted

    @property
    def stats(self) -> Dict[str, Tuple[float, float]]:
        """Return fitted statistics for inspection."""
        return dict(self._stats)

    def save(self, path: str) -> None:
        """Save normalization statistics to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "method": self.method,
                "stats": self._stats,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """Load normalization statistics from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        norm = cls(method=data["method"])
        norm._stats = {k: tuple(v) for k, v in data["stats"].items()}
        norm._fitted = len(norm._stats) > 0
        return norm
