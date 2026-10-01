"""
Test synthetic data generation.

Validates all requirements from the HITGNN Phase 2 specification:
    - Correct number of patients generated
    - Approximate mortality prevalence (~20%)
    - All event timestamps <= t0 (no future leakage)
    - No NaN in required fields
    - All event types present across cohort
    - Variable trajectory lengths
    - Missingness patterns exist
    - No duplicate patient IDs
    - Patient-level splits have no overlap
    - Repeated observations exist for selected biomarkers
    - At least some biomarkers are never observed for some patients
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.data.vocab import ConceptVocabulary, EVENT_TYPES
from src.data.terminology import ClinicalTerminology
from src.data.synthetic_generator import (
    SyntheticCohortGenerator,
    patient_level_split,
    save_cohort,
)
from src.data.canonical import ClinicalEvent, PatientRecord
from src.utils.config import load_config


# ---- Fixtures ------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    """Load default config."""
    return load_config("configs/default.yaml")


@pytest.fixture(scope="module")
def vocab():
    """Build default vocabulary."""
    return ConceptVocabulary.build_default()


@pytest.fixture(scope="module")
def cohort(config, vocab):
    """Generate a full synthetic cohort (cached for module scope)."""
    seed = config.get("seed", 42)
    gen = SyntheticCohortGenerator(config, vocab, seed=seed)
    return gen.generate()


@pytest.fixture(scope="module")
def splits(cohort, config):
    """Generate patient-level splits."""
    ratios = config.get("training", {}).get(
        "split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    ids = [p.patient_id for p in cohort]
    labels = [p.mortality for p in cohort]
    return patient_level_split(ids, ratios, seed=config.get("seed", 42), labels=labels)


# ---- Tests ---------------------------------------------------------------

class TestSyntheticGeneration:
    """Tests for SyntheticCohortGenerator."""

    def test_patient_count(self, cohort, config):
        """Should generate configured number of patients."""
        expected = config.get("synthetic", {}).get("n_patients", 500)
        assert len(cohort) == expected

    def test_mortality_labels_binary(self, cohort):
        """Mortality labels must be 0 or 1."""
        for p in cohort:
            assert p.mortality in (0, 1), (
                f"Patient {p.patient_id} has non-binary mortality: {p.mortality}"
            )

    def test_mortality_prevalence(self, cohort, config):
        """Mortality rate should be approximately 20% (within 8% absolute)."""
        target = config.get("synthetic", {}).get("mortality_prevalence", 0.20)
        actual = sum(p.mortality for p in cohort) / len(cohort)
        assert abs(actual - target) < 0.08, (
            f"Mortality prevalence {actual:.3f} deviates >8% from target {target}"
        )

    def test_no_future_events(self, cohort):
        """All event timestamps must be <= 0 (relative to t0)."""
        for p in cohort:
            for e in p.events:
                assert e.timestamp <= 0.0, (
                    f"Patient {p.patient_id} has event at t={e.timestamp} > t0"
                )

    def test_no_nan_required_fields(self, cohort):
        """patient_id, timestamp, event_type, concept_id must not be None."""
        for p in cohort:
            assert p.patient_id is not None
            assert p.metadata is not None
            for e in p.events:
                assert e.patient_id is not None
                assert e.timestamp is not None
                assert e.event_type is not None
                assert e.concept_id is not None
                assert e.concept_name is not None

    def test_all_event_types_present(self, cohort):
        """All six event types should appear across the cohort."""
        observed = set()
        for p in cohort:
            for e in p.events:
                observed.add(e.event_type)
        expected = set(EVENT_TYPES)
        assert observed == expected, (
            f"Missing event types: {expected - observed}"
        )

    def test_valid_event_types(self, cohort):
        """All events must have a valid event type."""
        valid = set(EVENT_TYPES)
        for p in cohort:
            for e in p.events:
                assert e.event_type in valid, (
                    f"Invalid event type: {e.event_type}"
                )

    def test_variable_trajectory_lengths(self, cohort):
        """Patients should have different numbers of events."""
        counts = [len(p.events) for p in cohort]
        unique_counts = len(set(counts))
        # With 500 patients, we expect many distinct event counts
        assert unique_counts > 20, (
            f"Only {unique_counts} distinct event counts — insufficient variation"
        )

    def test_missingness_exists(self, cohort):
        """Some biomarkers should be missing for some patients."""
        biomarkers = ["Lactate", "Creatinine", "Bilirubin", "Platelets",
                      "WBC", "CRP", "PCT"]
        n = len(cohort)
        for bm in biomarkers:
            n_with = sum(
                1 for p in cohort
                if any(e.concept_name == bm for e in p.events)
            )
            # At least some patients should be missing each biomarker
            assert n_with < n, (
                f"No missingness for {bm}: all {n} patients have measurements"
            )

    def test_never_observed_biomarkers(self, cohort):
        """PCT and CRP should be never-observed in a substantial fraction."""
        n = len(cohort)
        for sparse_bm in ["PCT", "CRP"]:
            n_with = sum(
                1 for p in cohort
                if any(e.concept_name == sparse_bm for e in p.events)
            )
            # PCT and CRP should be absent in at least 20% of patients
            frac_missing = 1.0 - n_with / n
            assert frac_missing > 0.20, (
                f"{sparse_bm} only missing in {frac_missing:.1%} of patients — "
                f"expected >20%"
            )

    def test_repeated_observations(self, cohort):
        """Selected biomarkers should have repeated measurements for some patients."""
        labs_to_check = ["Lactate", "Creatinine", "WBC"]
        for lab in labs_to_check:
            has_repeated = False
            for p in cohort:
                count = sum(1 for e in p.events if e.concept_name == lab)
                if count >= 2:
                    has_repeated = True
                    break
            assert has_repeated, (
                f"No patient has repeated {lab} measurements"
            )

    def test_no_duplicate_patient_ids(self, cohort):
        """No duplicate patient IDs."""
        ids = [p.patient_id for p in cohort]
        assert len(ids) == len(set(ids)), "Duplicate patient IDs detected"

    def test_events_reference_own_patient(self, cohort):
        """Every event's patient_id must match its parent record."""
        for p in cohort:
            for e in p.events:
                assert e.patient_id == p.patient_id, (
                    f"Event patient_id {e.patient_id} != record {p.patient_id}"
                )

    def test_events_sorted_chronologically(self, cohort):
        """Events within each patient should be sorted by timestamp."""
        for p in cohort[:50]:  # Check first 50 for speed
            timestamps = [e.timestamp for e in p.events]
            assert timestamps == sorted(timestamps), (
                f"Events not sorted for {p.patient_id}"
            )

    def test_lab_values_plausible(self, cohort):
        """Lab values should be within broad physiological ranges."""
        ranges = {
            "Lactate": (0.1, 25.0),
            "Creatinine": (0.1, 15.0),
            "Bilirubin": (0.05, 35.0),
            "Platelets": (1.0, 700.0),
            "WBC": (0.1, 60.0),
        }
        for p in cohort:
            for e in p.events:
                if e.concept_name in ranges and e.value is not None:
                    lo, hi = ranges[e.concept_name]
                    assert lo <= e.value <= hi, (
                        f"{e.concept_name} value {e.value} outside [{lo}, {hi}]"
                    )

    def test_severity_overlapping_distributions(self, cohort):
        """Survivors and non-survivors should have overlapping severity.

        This verifies that mortality is NOT deterministic from severity.
        """
        surv_sev = [p._generator_latent_severity for p in cohort if p.mortality == 0]
        mort_sev = [p._generator_latent_severity for p in cohort if p.mortality == 1]

        assert len(surv_sev) > 0 and len(mort_sev) > 0

        # Ranges should overlap
        surv_max = max(surv_sev)
        mort_min = min(mort_sev)
        assert surv_max > mort_min, (
            "No overlap between survivor and non-survivor severity distributions"
        )

    def test_metadata_completeness(self, cohort):
        """All patients should have complete metadata."""
        for p in cohort:
            m = p.metadata
            assert "age" in m and isinstance(m["age"], (int, float))
            assert "sex" in m and m["sex"] in ("M", "F")
            assert "icu_type" in m and m["icu_type"] in ("MICU", "SICU", "CCU")
            assert "sofa_score" in m and isinstance(m["sofa_score"], (int, float))
            assert "comorbidities" in m and isinstance(m["comorbidities"], dict)


class TestPatientSplits:
    """Tests for patient-level splitting."""

    def test_no_split_overlap(self, splits):
        """No patient may appear in more than one split."""
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert len(train_set & val_set) == 0, "Train/val overlap"
        assert len(train_set & test_set) == 0, "Train/test overlap"
        assert len(val_set & test_set) == 0, "Val/test overlap"

    def test_split_covers_all(self, splits, cohort):
        """All patients should be assigned to exactly one split."""
        all_split_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        all_patient_ids = {p.patient_id for p in cohort}
        assert all_split_ids == all_patient_ids

    def test_split_proportions(self, splits, cohort):
        """Split sizes should approximately match 70/15/15."""
        n = len(cohort)
        assert abs(len(splits["train"]) / n - 0.70) < 0.05
        assert abs(len(splits["val"]) / n - 0.15) < 0.05
        assert abs(len(splits["test"]) / n - 0.15) < 0.05

    def test_split_reproducibility(self, cohort, config):
        """Same seed should produce same splits."""
        ids = [p.patient_id for p in cohort]
        labels = [p.mortality for p in cohort]
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        seed = config.get("seed", 42)
        s1 = patient_level_split(ids, ratios, seed=seed, labels=labels)
        s2 = patient_level_split(ids, ratios, seed=seed, labels=labels)
        assert s1 == s2

    def test_stratified_prevalence(self, splits, cohort):
        """Each split should have mortality prevalence within +/-5pp of overall."""
        n = len(cohort)
        overall_prev = sum(p.mortality for p in cohort) / n
        pid_to_mort = {p.patient_id: p.mortality for p in cohort}

        for split_name, ids in splits.items():
            if len(ids) == 0:
                continue
            split_prev = sum(pid_to_mort[pid] for pid in ids) / len(ids)
            assert abs(split_prev - overall_prev) < 0.05, (
                f"{split_name} prevalence {split_prev:.3f} deviates >5pp "
                f"from overall {overall_prev:.3f}"
            )


class TestSeverityLeakage:
    """Ensure latent generator severity never reaches model-facing data."""

    def test_no_severity_in_patients_csv(self, cohort, config):
        """patients.csv must NOT contain 'severity' or '_generator_latent_severity'."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_cohort(cohort, output_dir=tmpdir)
            import pandas as pd
            df = pd.read_csv(paths["patients"])
            forbidden = {"severity", "latent_severity", "severity_latent"}
            found = forbidden & set(df.columns)
            assert len(found) == 0, (
                f"Latent severity leaked into patients.csv columns: {found}"
            )

    def test_model_facing_columns_allowed(self, cohort, config):
        """patients.csv must contain only observable EHR features."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_cohort(cohort, output_dir=tmpdir)
            import pandas as pd
            df = pd.read_csv(paths["patients"])
            allowed = {
                "patient_id", "age", "sex", "icu_type", "sofa_score",
                "mortality", "n_events",
                "diabetes", "CKD", "liver_disease", "heart_failure", "COPD",
            }
            unexpected = set(df.columns) - allowed
            assert len(unexpected) == 0, (
                f"Unexpected columns in patients.csv: {unexpected}"
            )


class TestVocabulary:
    """Tests for ConceptVocabulary.build_default()."""

    def test_default_vocab_size(self, vocab):
        """Default vocabulary should have 29 concepts."""
        assert vocab.size == 29

    def test_all_categories_present(self, vocab):
        """All six event type categories should be represented."""
        for cat in EVENT_TYPES:
            concepts = vocab.get_concepts_by_category(cat)
            assert len(concepts) > 0, f"No concepts in category {cat}"

    def test_lab_concepts_have_units(self, vocab):
        """Lab concepts should have units."""
        labs = vocab.get_concepts_by_category("LAB")
        for lab in labs:
            assert lab.unit is not None, f"Lab {lab.concept_name} has no unit"

    def test_lab_concepts_have_normal_range(self, vocab):
        """Lab concepts should have normal ranges."""
        labs = vocab.get_concepts_by_category("LAB")
        for lab in labs:
            assert lab.normal_range is not None, (
                f"Lab {lab.concept_name} has no normal range"
            )


class TestTerminology:
    """Tests for ClinicalTerminology.build_default()."""

    def test_default_terminology_relations(self):
        """Default terminology should have substantial relations."""
        t = ClinicalTerminology.build_default()
        assert len(t.all_relations) >= 40, (
            f"Only {len(t.all_relations)} relations — expected ~50"
        )

    def test_lactate_has_relations(self):
        """Lactate should have semantic relations."""
        t = ClinicalTerminology.build_default()
        rels = t.get_relations_for("Lactate")
        assert len(rels) >= 2

    def test_semantic_groups_assigned(self):
        """All vocab concepts should have semantic groups."""
        t = ClinicalTerminology.build_default()
        vocab = ConceptVocabulary.build_default()
        for cid in range(vocab.size):
            name = vocab.id_to_name(cid)
            group = t.get_semantic_group(name)
            assert group is not None, f"No semantic group for {name}"
