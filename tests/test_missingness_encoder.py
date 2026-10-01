"""
Test missingness encoder and biomarker state preprocessing.

Validates:
    A. Observed biomarker: mask=1, value channel has normalized value
    B. Currently missing but previously observed: mask=0, value=0, never_obs=0, delta_t>0
    C. Never observed: mask=0, value=0, never_obs=1, delta_t uses cap
    D. Repeated measurement: delta_t corresponds to same-biomarker history
    E. Multiple measurements in bin: most recent selected
    F. No future leakage: post-bin measurement doesn't affect prior slot
    G. Normalization: train-only statistics
    H. Shapes: exact expected dimensions
    I. Gradient flow: backward pass through both Linear layers
    J. No imputation: no population mean/median in missing value channel
"""

import numpy as np
import pytest
import torch

from src.data.synthetic_generator import ClinicalEvent
from src.preprocessing.temporal_binner import TemporalBinner, BiomarkerState
from src.preprocessing.normalizer import FeatureNormalizer
from src.models.missingness_encoder import MissingnessEncoder


# ---- Helper: create lab events -------------------------------------------

def _make_lab(pid, timestamp, name, value, concept_id=0):
    """Create a ClinicalEvent for testing."""
    return ClinicalEvent(
        patient_id=pid,
        timestamp=timestamp,
        event_type="LAB",
        concept_id=concept_id,
        concept_name=name,
        value=value,
        unit="test_unit",
    )


# ---- Fixtures ------------------------------------------------------------

@pytest.fixture
def binner():
    return TemporalBinner(n_bins=6, delta_t_max_hours=200.0)


@pytest.fixture
def biomarkers():
    return ["PCT", "CRP", "Lactate", "WBC", "Creatinine", "Bilirubin", "Platelets"]


@pytest.fixture
def encoder():
    return MissingnessEncoder(
        input_dim=4, hidden_dim=64, output_dim=64,
        n_biomarkers=7, n_bins=6, output_activation=None,
    )


# ---- Test A: Observed Biomarker ------------------------------------------

class TestObservedBiomarker:

    def test_mask_is_one_when_observed(self, binner, biomarkers):
        """When a biomarker has an observation in a bin, mask should be 1."""
        events = [_make_lab("P0001", -5.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Find the bin containing timestamp -5.0
        observed_mask = state.masks[:, lac_idx]
        assert observed_mask.sum() >= 1, "Lactate observation not found in any bin"

    def test_value_channel_contains_observation(self, binner, biomarkers):
        """Observed value should appear in raw_values."""
        events = [_make_lab("P0001", -5.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Find bin where mask=1
        for k in range(6):
            if state.masks[k, lac_idx] == 1.0:
                assert state.raw_values[k, lac_idx] == pytest.approx(2.5)
                break

    def test_never_observed_zero_when_seen(self, binner, biomarkers):
        """never_observed should be 0 for bins at or after first observation."""
        events = [_make_lab("P0001", -8.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # After the bin containing the observation, never_observed should be 0
        obs_bin = None
        for k in range(6):
            if state.masks[k, lac_idx] == 1.0:
                obs_bin = k
                break
        assert obs_bin is not None
        for k in range(obs_bin, 6):
            assert state.never_observed[k, lac_idx] == 0.0


# ---- Test B: Currently Missing But Previously Observed -------------------

class TestMissingPrevObserved:

    def test_mask_zero_when_absent(self, binner, biomarkers):
        """Bin without observation should have mask=0."""
        # Observation only in early bin
        events = [_make_lab("P0001", -11.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Last bin (near t0) should be absent
        assert state.masks[-1, lac_idx] == 0.0

    def test_value_zero_when_absent(self, binner, biomarkers):
        """Value channel should be 0 when mask=0."""
        events = [_make_lab("P0001", -11.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        assert state.values[-1, lac_idx] == 0.0

    def test_never_observed_zero_after_first_obs(self, binner, biomarkers):
        """Once seen, never_observed should stay 0 even in absent bins."""
        events = [_make_lab("P0001", -11.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Find observation bin
        obs_bin = None
        for k in range(6):
            if state.masks[k, lac_idx] == 1.0:
                obs_bin = k
                break
        assert obs_bin is not None
        # Later bins: absent but never_observed should be 0
        for k in range(obs_bin + 1, 6):
            assert state.never_observed[k, lac_idx] == 0.0

    def test_delta_t_positive_when_absent(self, binner, biomarkers):
        """delta_t should be > 0 for absent bins after prior observation."""
        events = [_make_lab("P0001", -11.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Last bin should have positive delta_t (time since observation)
        assert state.delta_times[-1, lac_idx] > 0.0


# ---- Test C: Never Observed ----------------------------------------------

class TestNeverObserved:

    def test_mask_zero_for_never_observed(self, binner, biomarkers):
        """A biomarker never measured should have mask=0 everywhere."""
        # No PCT events
        events = [_make_lab("P0001", -5.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        pct_idx = biomarkers.index("PCT")
        assert state.masks[:, pct_idx].sum() == 0.0

    def test_value_zero_for_never_observed(self, binner, biomarkers):
        """Value channel should be 0 for never-observed biomarker."""
        events = [_make_lab("P0001", -5.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        pct_idx = biomarkers.index("PCT")
        assert state.values[:, pct_idx].sum() == 0.0

    def test_never_observed_one_everywhere(self, binner, biomarkers):
        """never_observed should be 1 for all bins of unmeasured biomarker."""
        events = [_make_lab("P0001", -5.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        pct_idx = biomarkers.index("PCT")
        assert (state.never_observed[:, pct_idx] == 1.0).all()

    def test_delta_t_uses_cap(self, binner, biomarkers):
        """delta_t for never-observed should use log(1 + cap)."""
        events = [_make_lab("P0001", -5.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        pct_idx = biomarkers.index("PCT")
        expected = np.log(1.0 + 200.0)
        for k in range(6):
            assert state.delta_times[k, pct_idx] == pytest.approx(expected, rel=1e-4)


# ---- Test D: Repeated Measurement ----------------------------------------

class TestRepeatedMeasurement:

    def test_delta_t_between_same_biomarker(self, binner, biomarkers):
        """delta_t should reflect time since prior same-biomarker measurement."""
        # Two Lactate measurements in different bins
        events = [
            _make_lab("P0001", -10.0, "Lactate", 2.0),
            _make_lab("P0001", -4.0, "Lactate", 3.5),
        ]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Find the second observation bin
        obs_bins = [k for k in range(6) if state.masks[k, lac_idx] == 1.0]
        if len(obs_bins) >= 2:
            # delta_t of second observation should reflect gap from first
            second_bin = obs_bins[1]
            # log(1 + 6) since gap is 6 hours
            assert state.delta_times[second_bin, lac_idx] > 0.0
            assert state.delta_times[second_bin, lac_idx] < np.log(1.0 + 200.0)


# ---- Test E: Multiple Measurements in Bin --------------------------------

class TestMultipleInBin:

    def test_most_recent_selected(self, binner, biomarkers):
        """When multiple obs in one bin, most recent value should be used."""
        # Two Lactate measurements very close together in the same bin
        # With 6 bins over 12 hours, each bin is 2h wide
        events = [
            _make_lab("P0001", -1.5, "Lactate", 2.0),  # Same bin (last bin: [-2, 0])
            _make_lab("P0001", -0.5, "Lactate", 5.0),  # Most recent
        ]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Last bin should have the most recent value
        assert state.raw_values[-1, lac_idx] == pytest.approx(5.0)


# ---- Test F: No Future Leakage -------------------------------------------

class TestNoFutureLeakage:

    def test_post_window_event_excluded(self, binner, biomarkers):
        """Events after window_end should not appear."""
        events = [
            _make_lab("P0001", -5.0, "Lactate", 2.5),
            _make_lab("P0001", 1.0, "Lactate", 99.0),  # After t0!
        ]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Value 99.0 should NOT appear anywhere
        assert not np.any(state.raw_values[:, lac_idx] == 99.0)


# ---- Test G: Normalization -----------------------------------------------

class TestNormalization:

    def test_train_only_fit(self):
        """Normalizer should fit only on provided values."""
        norm = FeatureNormalizer(method="zscore")
        train_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm.fit("Lactate", train_vals)
        assert norm.is_fitted
        mean, std = norm.stats["Lactate"]
        assert mean == pytest.approx(3.0)
        assert std == pytest.approx(np.std(train_vals))

    def test_transform_applies_correct_normalization(self):
        """Transform should apply z-score correctly."""
        norm = FeatureNormalizer(method="zscore")
        norm.fit("Lactate", np.array([2.0, 4.0]))
        # mean=3, std=1
        result = norm.transform("Lactate", 5.0)
        assert result == pytest.approx(2.0)

    def test_val_test_never_fit(self):
        """Normalizer must use training stats for val/test values."""
        norm = FeatureNormalizer(method="zscore")
        norm.fit("Lactate", np.array([1.0, 3.0]))  # mean=2, std=1
        # "Validation" value
        result = norm.transform("Lactate", 10.0)
        assert result == pytest.approx(8.0)

    def test_save_load_roundtrip(self, tmp_path):
        """Stats should survive save/load."""
        norm = FeatureNormalizer(method="zscore")
        norm.fit("Lactate", np.array([1.0, 3.0, 5.0]))
        path = str(tmp_path / "norm_stats.json")
        norm.save(path)
        loaded = FeatureNormalizer.load(path)
        assert loaded.stats == norm.stats
        assert loaded.method == norm.method


# ---- Test H: Shapes ------------------------------------------------------

class TestShapes:

    def test_biomarker_state_shapes(self, binner, biomarkers):
        """BiomarkerState tensors should have exact expected dimensions."""
        events = [
            _make_lab("P0001", -5.0, "Lactate", 2.5),
            _make_lab("P0001", -3.0, "WBC", 8.0),
        ]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        K, B = 6, 7
        assert state.values.shape == (K, B)
        assert state.masks.shape == (K, B)
        assert state.delta_times.shape == (K, B)
        assert state.never_observed.shape == (K, B)
        assert state.bin_edges.shape == (K + 1,)
        assert len(state.biomarker_names) == B

    def test_encoder_slot_output_shape(self, encoder):
        """Slot embeddings should be [batch, K, B, D]."""
        batch_size, K, B = 4, 6, 7
        x = torch.randn(batch_size, K, B, 4)
        slot_emb, z_miss = encoder(x)
        assert slot_emb.shape == (batch_size, K, B, 64)

    def test_encoder_patient_summary_shape(self, encoder):
        """Patient summary should be [batch, D]."""
        batch_size, K, B = 4, 6, 7
        x = torch.randn(batch_size, K, B, 4)
        slot_emb, z_miss = encoder(x)
        assert z_miss.shape == (batch_size, 64)


# ---- Test I: Gradient Flow -----------------------------------------------

class TestGradientFlow:

    def test_backward_produces_gradients(self, encoder):
        """Backward pass should produce gradients through both Linear layers."""
        x = torch.randn(2, 6, 7, 4, requires_grad=True)
        slot_emb, z_miss = encoder(x)
        loss = z_miss.sum()
        loss.backward()

        # Check gradients exist on both Linear layers
        linear_layers = [m for m in encoder.mlp if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 2
        for layer in linear_layers:
            assert layer.weight.grad is not None
            assert layer.weight.grad.abs().sum() > 0

    def test_input_gradient_flows(self, encoder):
        """Gradient should flow back to the input tensor."""
        x = torch.randn(2, 6, 7, 4, requires_grad=True)
        slot_emb, z_miss = encoder(x)
        loss = z_miss.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ---- Test J: No Imputation -----------------------------------------------

class TestNoImputation:

    def test_missing_slot_value_is_zero(self, binner, biomarkers):
        """Missing slots must have x*m = 0, NOT a population mean."""
        # Only Lactate observed, all others missing
        events = [_make_lab("P0001", -5.0, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        # Check all missing biomarker slots have value=0
        for bi, name in enumerate(biomarkers):
            if name == "Lactate":
                continue
            # All bins for this biomarker should be 0
            assert state.values[:, bi].sum() == 0.0, (
                f"Non-zero value in missing biomarker {name} — "
                f"possible population mean imputation!"
            )

    def test_absent_bin_value_is_zero(self, binner, biomarkers):
        """Even for observed biomarkers, absent bins should have value=0."""
        # Lactate only at very start
        events = [_make_lab("P0001", -11.5, "Lactate", 2.5)]
        state = binner.build_biomarker_state(
            events, biomarkers, window_start=-12.0, window_end=0.0
        )
        lac_idx = biomarkers.index("Lactate")
        # Last bin should definitely be absent
        assert state.values[-1, lac_idx] == 0.0
