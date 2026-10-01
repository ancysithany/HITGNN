"""
Tests for the dataset adapters.
"""

from src.data.canonical import PatientRecord, ClinicalEvent, CanonicalCohort
from src.data.adapters.base import BaseDatasetAdapter
from src.data.adapters.synthetic import SyntheticDatasetAdapter
from src.data.factory import create_dataset_adapter

def test_synthetic_adapter_loads_correctly():
    adapter = create_dataset_adapter({"dataset": {"adapter": "synthetic"}})
    cohort = adapter.prepare()
    
    assert isinstance(cohort, CanonicalCohort)
    assert len(cohort.patients) == 500
    assert len(cohort.train_ids) == 349
    assert len(cohort.val_ids) == 74
    assert len(cohort.test_ids) == 77
    
    # Check one patient
    p = cohort.get_patient("P0000")
    assert p is not None
    assert p.age > 0
    assert p.sex in ["M", "F"]
    assert p.t0 == 0.0
    assert len(p.events) > 0
    assert p.events[0].source == "synthetic"

def test_dummy_adapter_integration():
    """Verify that a dummy adapter can be passed to the pipeline."""
    from src.preprocessing.pipeline import build_hitgnn_datasets
    
    class DummyAdapter(BaseDatasetAdapter):
        def prepare(self):
            events = [
                ClinicalEvent("DUMMY1", 0.0, "LAB", 1, "Lactate", 2.0, "mmol/L", "dummy")
            ]
            patients = [
                PatientRecord("DUMMY1", 50, "M", "MICU", 4.0, {"diabetes": 0}, 0.0, 1, split="train", events=events)
            ]
            return CanonicalCohort(patients, ["DUMMY1"], [], [])
            
        def load_patients(self): return []
        def load_events(self): return []
        def validate(self): pass
        
    adapter = DummyAdapter()
    cohort = adapter.prepare()
    
    train_ds, val_ds, test_ds, vocab = build_hitgnn_datasets(cohort, {"biomarkers": ["Lactate"]})
    assert len(train_ds) == 1
    assert len(val_ds) == 0
    assert train_ds[0].num_nodes > 0
