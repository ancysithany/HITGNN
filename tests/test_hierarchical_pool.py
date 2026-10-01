"""
tests/test_hierarchical_pool.py — Phase 6 Hierarchical Pooling Tests.

Tests 1-18 as specified in the Phase 6 specification.
"""

import torch
import pytest
from torch_geometric.data import Data, Batch

from src.models.hierarchical_pool import HierarchicalAttentionPool
from src.models.hitgnn import PatientEncoder
from src.data.vocab import ConceptVocabulary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
FUSED_DIM = 128
MISSINGNESS_DIM = 64
N_TYPES = 6
NUM_CONCEPTS = 29


def _make_graph(n_nodes: int = 8, n_edges: int = 12, seed: int = 0, missing_types: list[int] = None) -> Data:
    """Construct a small random patient graph with correct attribute types."""
    torch.manual_seed(seed)
    data = Data()
    
    # Generate node types, optionally ensuring some types are missing
    types = torch.randint(0, 6, (n_nodes,))
    if missing_types:
        for t in missing_types:
            types[types == t] = (t + 1) % 6
    data.node_type = types
    
    data.concept_id = torch.randint(0, NUM_CONCEPTS, (n_nodes,))
    data.numeric_value = torch.randn(n_nodes)
    data.has_value = (torch.rand(n_nodes) > 0.3).float()
    data.relative_time = -torch.rand(n_nodes) * 48.0
    data.time = data.relative_time.clone()

    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    if src.size(0) == 0:
        # Fallback if all removed
        src = torch.tensor([0, 1])
        dst = torch.tensor([1, 0])
        
    data.edge_index = torch.stack([src, dst], dim=0)
    E = src.size(0)
    data.edge_delta_t = torch.rand(E) * 20.0
    data.edge_type = torch.randint(0, 4, (E,))
    return data


def _make_batch(num_graphs: int = 4, missing_types: list[int] = None) -> Batch:
    graphs = [_make_graph(n_nodes=10+i, n_edges=15+i, seed=i, missing_types=missing_types) for i in range(num_graphs)]
    batch = Batch.from_data_list(graphs)
    # Add dummy missingness input
    batch.missingness_input = torch.rand(num_graphs, 6, 7, 4)
    return batch


def _make_pool() -> HierarchicalAttentionPool:
    return HierarchicalAttentionPool(
        hidden_dim=HIDDEN_DIM,
        n_types=N_TYPES,
        missingness_dim=MISSINGNESS_DIM,
        fused_dim=FUSED_DIM,
        dropout=0.0
    )


# ---------------------------------------------------------------------------
# Test 1-4 — Output Shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_shapes(self):
        batch = _make_batch(num_graphs=3)
        h_nodes = torch.randn(batch.num_nodes, HIDDEN_DIM)
        z_miss = torch.randn(3, MISSINGNESS_DIM)
        
        pool = _make_pool()
        fused, h_p, h_tau, mask, beta, alpha = pool(
            h=h_nodes,
            node_type=batch.node_type,
            batch_index=batch.batch,
            batch_size=batch.num_graphs,
            z_miss=z_miss
        )
        
        # TEST 1
        assert h_tau.shape == (3, 6, 64)
        # TEST 2
        assert h_p.shape == (3, 64)
        # TEST 3 (Missingness summary shape tested in PatientEncoder)
        # TEST 4
        assert fused.shape == (3, 128)


# ---------------------------------------------------------------------------
# Test 5-6 — Attention Normalisation
# ---------------------------------------------------------------------------

class TestAttentionNormalisation:
    def test_type_level_sums_to_one(self):
        """TEST 5: type-level attention sums to 1 per patient/type."""
        batch = _make_batch(num_graphs=2)
        h_nodes = torch.randn(batch.num_nodes, HIDDEN_DIM)
        z_miss = torch.randn(2, MISSINGNESS_DIM)
        pool = _make_pool()
        
        _, _, _, mask, _, alpha = pool(
            h=h_nodes, node_type=batch.node_type, batch_index=batch.batch,
            batch_size=batch.num_graphs, z_miss=z_miss
        )
        
        # Verify sum per (patient, type)
        for p in range(batch.num_graphs):
            for t in range(6):
                if mask[p, t] == 1.0:
                    idx = (batch.batch == p) & (batch.node_type == t)
                    s = alpha[idx].sum()
                    assert torch.allclose(s, torch.ones(1), atol=1e-4)

    def test_patient_level_sums_to_one(self):
        """TEST 6: patient-level type attention sums to 1 over present types."""
        batch = _make_batch(num_graphs=2)
        h_nodes = torch.randn(batch.num_nodes, HIDDEN_DIM)
        z_miss = torch.randn(2, MISSINGNESS_DIM)
        pool = _make_pool()
        
        _, _, _, mask, beta, _ = pool(
            h=h_nodes, node_type=batch.node_type, batch_index=batch.batch,
            batch_size=batch.num_graphs, z_miss=z_miss
        )
        
        sums = beta.sum(dim=1)
        # They should all sum to 1
        assert torch.allclose(sums, torch.ones(2), atol=1e-4)


# ---------------------------------------------------------------------------
# Test 7, 9 — Absent Event Types
# ---------------------------------------------------------------------------

class TestAbsentEventTypes:
    def test_absent_types_zero_attention_and_finite(self):
        """TEST 7 & 9: absent event types receive zero attention, no NaN/Inf."""
        # Graph with NO nodes of type 2 and 3
        batch = _make_batch(num_graphs=2, missing_types=[2, 3])
        h_nodes = torch.randn(batch.num_nodes, HIDDEN_DIM)
        z_miss = torch.randn(2, MISSINGNESS_DIM)
        pool = _make_pool()
        
        fused, h_p, _, mask, beta, _ = pool(
            h=h_nodes, node_type=batch.node_type, batch_index=batch.batch,
            batch_size=batch.num_graphs, z_miss=z_miss
        )
        
        assert mask[0, 2] == 0.0
        assert mask[0, 3] == 0.0
        assert beta[0, 2] == 0.0
        assert beta[0, 3] == 0.0
        assert torch.isfinite(fused).all()
        assert torch.isfinite(h_p).all()


# ---------------------------------------------------------------------------
# Test 8 — Single Node Type
# ---------------------------------------------------------------------------

class TestSingleNodeType:
    def test_single_node_attention(self):
        """TEST 8: single node type produces node attention = 1."""
        batch = _make_batch(num_graphs=1)
        # Force exactly one node to be type 0
        batch.node_type[1:] = 1
        batch.node_type[0] = 0
        
        h_nodes = torch.randn(batch.num_nodes, HIDDEN_DIM)
        z_miss = torch.randn(1, MISSINGNESS_DIM)
        pool = _make_pool()
        
        _, _, _, _, _, alpha = pool(
            h=h_nodes, node_type=batch.node_type, batch_index=batch.batch,
            batch_size=batch.num_graphs, z_miss=z_miss
        )
        
        # Node 0 is the only type 0 node, so its attention must be 1.0
        assert torch.allclose(alpha[0], torch.ones(1), atol=1e-4)


# ---------------------------------------------------------------------------
# Test 10 — Batch Isolation
# ---------------------------------------------------------------------------

class TestBatchIsolation:
    def test_patient_isolation_across_batch(self):
        """TEST 10: changing nodes in patient A does not change patient B representation."""
        batch = _make_batch(num_graphs=2)
        h_nodes = torch.randn(batch.num_nodes, HIDDEN_DIM)
        z_miss = torch.randn(2, MISSINGNESS_DIM)
        pool = _make_pool()
        pool.eval()
        
        with torch.no_grad():
            fused_1, _, _, _, _, _ = pool(
                h=h_nodes, node_type=batch.node_type, batch_index=batch.batch,
                batch_size=batch.num_graphs, z_miss=z_miss
            )
            
            # Modify patient 0's embeddings
            h_nodes_mod = h_nodes.clone()
            idx_p0 = (batch.batch == 0)
            h_nodes_mod[idx_p0] += 5.0
            
            fused_2, _, _, _, _, _ = pool(
                h=h_nodes_mod, node_type=batch.node_type, batch_index=batch.batch,
                batch_size=batch.num_graphs, z_miss=z_miss
            )
            
        # Patient 1's embedding should be perfectly identical
        assert torch.allclose(fused_1[1], fused_2[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 11 — Permutation Invariance
# ---------------------------------------------------------------------------

class TestPermutationInvariance:
    def test_invariance_to_node_ordering(self):
        """TEST 11: permutation invariance."""
        batch = _make_batch(num_graphs=1)
        h_nodes = torch.randn(batch.num_nodes, HIDDEN_DIM)
        z_miss = torch.randn(1, MISSINGNESS_DIM)
        pool = _make_pool()
        pool.eval()
        
        with torch.no_grad():
            fused_orig, _, _, _, _, _ = pool(
                h=h_nodes, node_type=batch.node_type, batch_index=batch.batch,
                batch_size=batch.num_graphs, z_miss=z_miss
            )
            
            # Shuffle nodes
            perm = torch.randperm(batch.num_nodes)
            h_nodes_shuff = h_nodes[perm]
            node_type_shuff = batch.node_type[perm]
            batch_index_shuff = batch.batch[perm]
            
            fused_shuff, _, _, _, _, _ = pool(
                h=h_nodes_shuff, node_type=node_type_shuff, batch_index=batch_index_shuff,
                batch_size=batch.num_graphs, z_miss=z_miss
            )
            
        assert torch.allclose(fused_orig, fused_shuff, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 12-18 — Patient Encoder (Full forward + gradients)
# ---------------------------------------------------------------------------

class TestPatientEncoder:
    def test_full_encoder_forward_and_gradients(self):
        """TEST 12, 13, 14, 15, 16, 17, 18: Full PatientEncoder testing."""
        batch = _make_batch(num_graphs=2)
        config = {
            "te_gat": {"hidden_dim": 64, "n_heads": 4, "n_layers": 3, "time_encoding_dim": 16, "edge_type_dim": 8},
            "missingness": {"hidden_dim": 64},
            "pooling": {"fused_dim": 128, "dropout": 0.0}
        }
        encoder = PatientEncoder(config, vocab_size=NUM_CONCEPTS)
        encoder.train()
        
        # Forward pass (TEST 18)
        out = encoder(batch, return_attention=True)
        
        # Assertions
        assert out.fused_embedding.shape == (2, 128)
        
        # TEST 3: missingness_summary shape
        assert out.missingness_summary.shape == (2, 64)
        
        # TEST 15: slot embeddings preserved
        assert out.slot_missingness_embeddings.shape == (2, 6, 7, 64)
        
        # TEST 16: TE-GAT attention retrievable
        assert out.te_gat_attention is not None
        assert len(out.te_gat_attention) == 3
        
        # TEST 17: node type attention aligned with total nodes
        assert out.node_attention.shape == (batch.num_nodes,)
        
        # Backward pass
        loss = out.fused_embedding.mean()
        loss.backward()
        
        # Verify gradients exist throughout the network (TEST 12, 13, 14)
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"
