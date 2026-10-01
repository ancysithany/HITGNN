"""
src.preprocessing.temporal_binner — Temporal binning for missingness encoder.

Divides each patient's observation window [obs_start, t0] into K
equal-width bins and assigns laboratory measurements to bins.

Bin boundary convention:
    [left_edge, right_edge)  for bins 0..K-2
    [left_edge, right_edge]  for the FINAL bin (includes t0 exactly)

When multiple observations of the same biomarker fall in one bin,
the MOST RECENT measurement is selected.

Delta-t convention:
    delta_t_bk = time since the most recent PRIOR observation of the
    SAME biomarker (NOT time-since-t0, NOT time-since-any-event).
    Measured in HOURS.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---- BiomarkerState dataclass -------------------------------------------

@dataclass
class BiomarkerState:
    """Fixed-vocabulary biomarker state for a single patient.

    All tensors have shape [K, B] where K=temporal bins, B=biomarkers.

    Attributes
    ----------
    patient_id : str
        Patient identifier.
    values : np.ndarray
        Normalized observed values (x * mask). Shape [K, B].
    masks : np.ndarray
        Observation mask (1 if observed in bin). Shape [K, B].
    delta_times : np.ndarray
        log(1 + delta_t) where delta_t is hours since most recent
        prior same-biomarker observation. Shape [K, B].
    never_observed : np.ndarray
        1 if biomarker has NEVER been observed up to and including
        this bin; 0 once seen at least once. Shape [K, B].
    bin_edges : np.ndarray
        Bin boundary timestamps in hours relative to t0.
        Shape [K + 1].
    biomarker_names : List[str]
        Ordered biomarker names matching the B axis.
    raw_values : np.ndarray
        Unnormalized observed values (for debugging). Shape [K, B].
    """
    patient_id: str
    values: np.ndarray        # [K, B]
    masks: np.ndarray         # [K, B]
    delta_times: np.ndarray   # [K, B]
    never_observed: np.ndarray  # [K, B]
    bin_edges: np.ndarray     # [K + 1]
    biomarker_names: List[str] = field(default_factory=list)
    raw_values: np.ndarray = field(default_factory=lambda: np.array([]))


class TemporalBinner:
    """Assign clinical events to temporal bins for the missingness encoder.

    Parameters
    ----------
    n_bins : int
        Number of temporal bins (default from config: 6).
    delta_t_max_hours : float
        Cap for delta_t when a biomarker has never been observed.
        Used as a sentinel indicating "no prior observation".
    """

    def __init__(
        self,
        n_bins: int = 6,
        delta_t_max_hours: float = 200.0,
    ) -> None:
        self.n_bins = n_bins
        self.delta_t_max_hours = delta_t_max_hours

    def compute_bin_edges(
        self,
        window_start: float,
        window_end: float = 0.0,
    ) -> np.ndarray:
        """Compute bin edges for a patient's observation window.

        Parameters
        ----------
        window_start : float
            Start of observation window (hours relative to t0, negative).
        window_end : float
            End of observation window (0.0 = t0).

        Returns
        -------
        np.ndarray
            Array of bin edges of shape ``[n_bins + 1]``.
        """
        return np.linspace(window_start, window_end, self.n_bins + 1)

    def build_biomarker_state(
        self,
        events: list,
        biomarker_names: List[str],
        window_start: float,
        window_end: float = 0.0,
        normalizer=None,
    ) -> BiomarkerState:
        """Build the complete biomarker state for one patient.

        Parameters
        ----------
        events : list
            List of ClinicalEvent objects for this patient.
            Only LAB events matching biomarker_names are used.
        biomarker_names : List[str]
            Ordered list of biomarkers (defines axis B).
        window_start : float
            Start of observation window in hours relative to t0.
        window_end : float
            End of window (default 0.0 = t0).
        normalizer : FeatureNormalizer, optional
            If provided, observed values are normalized using fitted stats.

        Returns
        -------
        BiomarkerState
            Complete fixed-vocabulary biomarker state.
        """
        K = self.n_bins
        B = len(biomarker_names)
        bin_edges = self.compute_bin_edges(window_start, window_end)

        # Initialize output arrays
        values = np.zeros((K, B), dtype=np.float32)
        raw_values = np.zeros((K, B), dtype=np.float32)
        masks = np.zeros((K, B), dtype=np.float32)
        delta_times = np.zeros((K, B), dtype=np.float32)
        never_observed = np.ones((K, B), dtype=np.float32)  # Start as never-observed

        # Build name-to-index map
        bm_to_idx = {name: i for i, name in enumerate(biomarker_names)}

        # Collect lab events per biomarker, sorted by timestamp
        bm_events: Dict[int, List[Tuple[float, float]]] = {
            i: [] for i in range(B)
        }
        patient_id = None
        for e in events:
            if e.event_type != "LAB":
                continue
            if e.concept_name not in bm_to_idx:
                continue
            if e.value is None:
                continue
            # Enforce temporal boundary
            if e.timestamp > window_end:
                continue
            if e.timestamp < window_start:
                continue
            bi = bm_to_idx[e.concept_name]
            bm_events[bi].append((e.timestamp, float(e.value)))
            patient_id = e.patient_id

        # Sort each biomarker's events chronologically
        for bi in range(B):
            bm_events[bi].sort(key=lambda x: x[0])

        # --- Fill each bin for each biomarker ---
        for bi in range(B):
            observations = bm_events[bi]

            if len(observations) == 0:
                # Biomarker NEVER observed for this patient
                # never_observed stays 1, mask stays 0, value stays 0
                delta_times[:, bi] = np.log(1.0 + self.delta_t_max_hours)
                continue

            # Track most recent observation timestamp BEFORE current bin
            last_obs_time: Optional[float] = None
            obs_idx = 0  # Pointer into sorted observations

            for k in range(K):
                left = bin_edges[k]
                right = bin_edges[k + 1]

                # Find observations in this bin
                # Convention: [left, right) for bins 0..K-2; [left, right] for bin K-1
                bin_obs: List[Tuple[float, float]] = []
                while obs_idx < len(observations):
                    t, v = observations[obs_idx]
                    if k < K - 1:
                        # [left, right)
                        if t < left:
                            # Before this bin — still update last_obs_time
                            # Actually this shouldn't happen since we iterate forward
                            obs_idx += 1
                            continue
                        if t >= right:
                            break
                    else:
                        # Final bin: [left, right] (includes t0)
                        if t < left:
                            obs_idx += 1
                            continue
                        if t > right:
                            break
                    bin_obs.append((t, v))
                    obs_idx += 1

                # Has this biomarker been seen AT ALL up to now?
                # (including current bin observations)
                ever_seen_before = last_obs_time is not None
                has_current_obs = len(bin_obs) > 0

                if has_current_obs:
                    # Use MOST RECENT measurement in this bin
                    most_recent_t, most_recent_v = bin_obs[-1]

                    raw_values[k, bi] = most_recent_v
                    if normalizer is not None:
                        values[k, bi] = normalizer.transform(
                            biomarker_names[bi], most_recent_v
                        )
                    else:
                        values[k, bi] = most_recent_v
                    masks[k, bi] = 1.0

                    # Delta-t: time since prior same-biomarker observation
                    if ever_seen_before and last_obs_time is not None:
                        dt = most_recent_t - last_obs_time
                        delta_times[k, bi] = np.log(1.0 + max(dt, 0.0))
                    elif len(bin_obs) > 1:
                        # Multiple obs in this bin, use gap within bin
                        dt = most_recent_t - bin_obs[0][0]
                        delta_times[k, bi] = np.log(1.0 + max(dt, 0.0))
                    else:
                        # First-ever observation
                        delta_times[k, bi] = np.log(1.0 + self.delta_t_max_hours)

                    # Mark as seen
                    never_observed[k, bi] = 0.0
                    last_obs_time = most_recent_t

                else:
                    # No observation in this bin
                    masks[k, bi] = 0.0
                    values[k, bi] = 0.0  # x * m = 0
                    raw_values[k, bi] = 0.0

                    if ever_seen_before and last_obs_time is not None:
                        # Previously observed but absent now
                        never_observed[k, bi] = 0.0
                        bin_ref = (left + right) / 2.0  # Bin center
                        dt = bin_ref - last_obs_time
                        delta_times[k, bi] = np.log(1.0 + max(dt, 0.0))
                    else:
                        # Still never observed
                        never_observed[k, bi] = 1.0
                        delta_times[k, bi] = np.log(1.0 + self.delta_t_max_hours)

                # Update ever_seen across bins
                if has_current_obs and last_obs_time is None:
                    # We already set last_obs_time above
                    pass

        # Apply mask to values: value channel = x_normalized * mask
        values = values * masks

        return BiomarkerState(
            patient_id=patient_id or "",
            values=values,
            masks=masks,
            delta_times=delta_times,
            never_observed=never_observed,
            bin_edges=bin_edges,
            biomarker_names=list(biomarker_names),
            raw_values=raw_values,
        )
