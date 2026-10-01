"""
src.preprocessing.event_normalizer — Normalizer for continuous event features in the temporal event graph.

Provides z-score and min-max normalization for continuous clinical events
such as LABs and VITALs. This is kept separate from the Phase 3 FeatureNormalizer
so the graph pipeline does not interfere with the missingness pipeline.

Statistics MUST be computed on the TRAINING split only.
"""

import json
import os
from typing import Dict, Tuple

import numpy as np


class EventNormalizer:
    """Fit-transform normalizer for continuous clinical events in graphs.

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
            Clinical concept name (e.g., ``"Lactate"``, ``"Heart Rate"``).
        values : np.ndarray
            Array of observed training values.
        """
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            self._stats[concept_name] = (0.0, 1.0)
            return

        if self.method == "zscore":
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std < 1e-8:
                std = 1.0
            self._stats[concept_name] = (mean, std)
        elif self.method == "minmax":
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            rng = vmax - vmin
            if rng < 1e-8:
                rng = 1.0
            self._stats[concept_name] = (vmin, rng)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True

    def transform(self, concept_name: str, value: float) -> float:
        """Normalize a single value.

        If the concept has no fitted statistics, returns the raw value.
        """
        if concept_name not in self._stats:
            return float(value)

        if self.method == "zscore":
            mean, std = self._stats[concept_name]
            return float((value - mean) / std)
        elif self.method == "minmax":
            vmin, rng = self._stats[concept_name]
            return float((value - vmin) / rng)
        return float(value)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def stats(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._stats)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "method": self.method,
                "stats": self._stats,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EventNormalizer":
        with open(path) as f:
            data = json.load(f)
        norm = cls(method=data["method"])
        norm._stats = {k: tuple(v) for k, v in data["stats"].items()}
        norm._fitted = len(norm._stats) > 0
        return norm
