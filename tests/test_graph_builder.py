"""
Test graph construction.

Validates Phase 4 constraints strictly.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.data.synthetic_generator import ClinicalEvent, PatientRecord
from src.data.terminology import ClinicalTerminology
from src.data.vocab import ConceptVocabulary
from src.preprocessing.event_normalizer import EventNormalizer
from src.graph.graph_builder import (
    PatientGraphBuilder, validate_patient_graph,
    EDGE_TYPE_TEMPORAL_ADJACENCY, EDGE_TYPE_SAME_CONCEPT,
    EDGE_TYPE_SEMANTIC, EDGE_TYPE_CROSS_MODALITY
)

@pytest.fixture
def builder():
    vocab = ConceptVocabulary.build_default()
    term = ClinicalTerminology.build_default()
    norm = EventNormalizer(method="zscore")
    config = {"cross_modality_window_min": 30.0}
    return PatientGraphBuilder(vocab, term, norm, config)

def _make_event(t, etype, cname, val=None):
    return ClinicalEvent(
        patient_id="P0", timestamp=t, event_type=etype,
        concept_id=hash(cname) % 1000, concept_name=cname, value=val, unit=""
    )

def _make_patient(pid, events):
    return PatientRecord(
        patient_id=pid,
        events=events,
        mortality=0,
        age=50.0,
        sex="M",
        icu_type="MICU",
        sofa_score=2.0,
        comorbidities={},
        t0=0.0
    )

class TestGraphBuilder:
    """Tests for PatientGraphBuilder."""

    def test_node_creation(self, builder):
        """TEST 1: One observed event -> exactly one node."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0)
        ])
        g = builder.build(p)
        assert g.node_type.size(0) == 1
        assert g.numeric_value.size(0) == 1

    def test_no_fake_missing_nodes(self, builder):
        """TEST 2: Patient never measured PCT -> zero PCT graph nodes."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0)
        ])
        g = builder.build(p)
        pct_id = hash("PCT") % 1000
        assert (g.concept_id == pct_id).sum().item() == 0

    def test_temporal_adjacency(self, builder):
        """TEST 3: Correct consecutive edges within category."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(-4.0, "LAB", "CRP", 10.0),
            _make_event(-3.0, "VITAL", "Heart Rate", 90),
        ])
        g = builder.build(p)
        # Should have 1 temporal adjacency edge: Lactate -> CRP
        mask = g.edge_type == EDGE_TYPE_TEMPORAL_ADJACENCY
        assert mask.sum().item() == 1
        dt = g.edge_delta_t[mask]
        assert dt.item() == 1.0

    def test_same_concept_recurrence(self, builder):
        """TEST 4: Repeated Lactate connects only consecutive Lactate."""
        p = _make_patient("P0", [
            _make_event(-10.0, "LAB", "Lactate", 2.0),
            _make_event(-8.0, "LAB", "Lactate", 2.5),
            _make_event(-2.0, "LAB", "Lactate", 3.0),
        ])
        g = builder.build(p)
        mask = g.edge_type == EDGE_TYPE_SAME_CONCEPT
        assert mask.sum().item() == 2
        dts = g.edge_delta_t[mask]
        # Should be sorted chronologically
        assert list(dts.numpy()) == [2.0, 6.0]

    def test_semantic_edge(self, builder):
        """TEST 5: Both related event concepts present -> semantic edge exists (temporally nearest)."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(-4.0, "CLINICAL_CONCEPT", "Tissue Hypoxia"),
        ])
        g = builder.build(p)
        mask = g.edge_type == EDGE_TYPE_SEMANTIC
        assert mask.sum().item() == 1
        assert g.edge_delta_t[mask].item() == 1.0

    def test_no_ontology_hallucination(self, builder):
        """TEST 6: Semantic target concept absent -> do not create graph node."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
        ])
        g = builder.build(p)
        # Only 1 node should exist
        assert g.node_type.size(0) == 1

    def test_cross_modality(self, builder):
        """TEST 7: Within 30 min -> edge exists. Outside -> no edge."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(-4.8, "CLINICAL_CONCEPT", "Sepsis"), # 12 mins -> edge
            _make_event(-3.0, "CLINICAL_CONCEPT", "Sepsis"), # 120 mins -> no edge from Lactate
        ])
        g = builder.build(p)
        mask = g.edge_type == EDGE_TYPE_CROSS_MODALITY
        # Only 1 cross-modality edge from Lactate to the nearest concept (at -4.8)
        assert mask.sum().item() == 1
        assert g.edge_delta_t[mask].item() == pytest.approx(0.2)

    def test_nonnegative_delta_t(self, builder):
        """TEST 8: All edge_delta_t >= 0."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(-4.8, "CLINICAL_CONCEPT", "Sepsis"),
        ])
        g = builder.build(p)
        assert (g.edge_delta_t >= 0).all()

    def test_parallel_edge_types(self, builder):
        """TEST 9: Same node pair can retain temporal + recurrence edges."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(-4.0, "LAB", "Lactate", 2.5),
        ])
        g = builder.build(p)
        # Should have 1 temporal adjacency and 1 same-concept recurrence between the same two nodes
        assert g.edge_index.size(1) == 2
        assert set(g.edge_type.numpy()) == {EDGE_TYPE_TEMPORAL_ADJACENCY, EDGE_TYPE_SAME_CONCEPT}

    def test_exact_duplicate_prevention(self, builder):
        """TEST 10: No repeated (source, destination, type)."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(-5.0, "LAB", "Lactate", 2.0), # duplicate
        ])
        g = builder.build(p)
        # Builder sorts and then connects. It will create duplicates initially but they must be deduplicated
        # There should be 1 temporal adjacency, 1 recurrence.
        hash_val = g.edge_index[0] * 1000 + g.edge_index[1] * 10 + g.edge_type
        assert len(torch.unique(hash_val)) == g.edge_index.size(1)

    def test_no_self_loops(self, builder):
        """TEST 11: No self loops."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
        ])
        g = builder.build(p)
        if g.edge_index.size(1) > 0:
            assert (g.edge_index[0] != g.edge_index[1]).all()

    def test_future_leakage(self, builder):
        """TEST 12: timestamp > t0 must be filtered."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(2.0, "LAB", "Lactate", 2.5),
        ])
        g = builder.build(p)
        assert g.node_type.size(0) == 1

    def test_label_isolation(self, builder):
        """TEST 13: mortality not inside node/edge metadata features."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
        ])
        g = builder.build(p)
        assert not hasattr(g, 'mortality')
        assert not hasattr(g, 'severity')

    def test_train_only_normalization(self, builder):
        """TEST 14: validation/test not used to fit event scaler."""
        # Not fully testable purely in builder without dataset context, but normalizer acts passively
        assert not builder.normalizer.is_fitted
        builder.normalizer.fit("Lactate", np.array([10.0, 20.0]))
        assert builder.normalizer.is_fitted

    def test_pyg_batching(self, builder):
        """TEST 15: multiple patient graphs batch correctly."""
        p1 = _make_patient("P1", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
        ])
        p2 = _make_patient("P2", [
            _make_event(-3.0, "LAB", "CRP", 10.0),
            _make_event(-2.0, "LAB", "CRP", 15.0),
        ])
        g1 = builder.build(p1)
        g2 = builder.build(p2)
        
        # We manually test DataBatch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([g1, g2])
        assert batch.num_nodes == 3
        # batching shifts indices correctly
        if batch.edge_index.size(1) > 0:
            assert batch.edge_index.max().item() < 3

    def test_validation_function(self, builder):
        """Validation function doesn't crash on good graphs."""
        p = _make_patient("P0", [
            _make_event(-5.0, "LAB", "Lactate", 2.0),
            _make_event(-4.0, "CLINICAL_CONCEPT", "Tissue Hypoxia"),
        ])
        g = builder.build(p)
        validate_patient_graph(g)
