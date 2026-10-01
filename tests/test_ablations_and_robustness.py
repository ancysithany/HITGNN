import pytest
import torch
import numpy as np

from src.data.synthetic_generator import SyntheticCohortGenerator
from src.data.vocab import ConceptVocabulary
from src.data.adapters.synthetic import SyntheticDatasetAdapter
from src.preprocessing.pipeline import build_hitgnn_datasets
from src.models.hitgnn import PatientMortalityModel
from torch_geometric.data import Batch

@pytest.fixture(scope="module")
def cohort():
    from src.data.synthetic_generator import save_cohort
    import tempfile
    
    vocab = ConceptVocabulary.build_default()
    gen = SyntheticCohortGenerator({"synthetic": {"n_patients": 10}}, vocab, seed=42)
    records = gen.generate()
    
    tmpdir = tempfile.mkdtemp()
    save_cohort(records, tmpdir)
    import json
    with open(f"{tmpdir}/splits.json", "w") as f:
        json.dump({"train": [p.patient_id for p in records], "val": [], "test": []}, f)
    adapter = SyntheticDatasetAdapter(tmpdir)
    return adapter.prepare()

def test_no_temporal_edges(cohort):
    config = {
        "graph": {
            "ablation": {"no_temporal_edges": True}
        }
    }
    train_ds, _, _, _ = build_hitgnn_datasets(cohort, config)
    data = train_ds[0]
    
    # 0 = TEMPORAL_ADJACENCY, 1 = SAME_CONCEPT
    assert not (data.edge_type == 0).any()
    assert not (data.edge_type == 1).any()

def test_no_semantic_edges(cohort):
    config = {
        "graph": {
            "ablation": {"no_semantic_edges": True}
        }
    }
    train_ds, _, _, _ = build_hitgnn_datasets(cohort, config)
    data = train_ds[0]
    
    # 2 = SEMANTIC
    assert not (data.edge_type == 2).any()

def test_no_missingness(cohort):
    config = {
        "graph": {},
        "ablation": {"no_missingness": True}
    }
    train_ds, _, _, vocab = build_hitgnn_datasets(cohort, config)
    data = train_ds[0]
    
    model = PatientMortalityModel(config, vocab.size)
    batch = Batch.from_data_list([data])
    out = model.encoder(batch)
    
    # Should not crash and should return values
    assert out.fused_embedding is not None

def test_mean_pooling(cohort):
    config = {
        "graph": {},
        "ablation": {"mean_pooling": True}
    }
    train_ds, _, _, vocab = build_hitgnn_datasets(cohort, config)
    data = train_ds[0]
    
    model = PatientMortalityModel(config, vocab.size)
    batch = Batch.from_data_list([data])
    out = model.encoder(batch)
    
    # Attention values should be None
    assert out.node_attention is None
    assert out.type_attention is None

def test_static_gnn(cohort):
    config = {
        "graph": {},
        "ablation": {"static_gnn": True}
    }
    train_ds, _, _, vocab = build_hitgnn_datasets(cohort, config)
    data = train_ds[0]
    
    model = PatientMortalityModel(config, vocab.size)
    batch = Batch.from_data_list([data])
    out = model.encoder(batch)
    assert out.fused_embedding is not None

def test_robustness_masking(cohort):
    config = {"graph": {}}
    train_ds, _, _, _ = build_hitgnn_datasets(cohort, config)
    
    # Base dataset without masking
    data_unmasked = train_ds[0]
    
    # Dataset with masking
    train_ds.masking_config = {"scenario": "MCAR", "rate": 1.0, "seed": 42}
    data_masked = train_ds[0]
    
    # The event graph should be identical!
    assert torch.equal(data_unmasked.edge_index, data_masked.edge_index)
    assert torch.equal(data_unmasked.node_type, data_masked.node_type)
    
    # The missingness state should be different (everything masked)
    assert (data_masked.biomarker_masks == 0).all()
    assert (data_masked.biomarker_values == 0).all()
    # never observed should be 1
    assert (data_masked.biomarker_never_observed == 1).all()
