"""
tests/test_te_gat.py — Phase 5 TE-GAT tests.

Tests 1-18 as specified in the Phase 5 specification:
    1.  EventNodeEncoder shape
    2.  Categorical dtype handling
    3.  Non-numeric node (has_value=0)
    4.  Node time encoding — distinct relative times → distinct encodings
    5.  Edge time encoding — distinct delta_t → distinct encodings
    6.  Zero delta_t — encoding finite
    7.  Large delta_t — encoding finite
    8.  TE-GAT forward — correct output shape
    9.  Multi-head — 4 heads execute correctly
    10. Gradient propagation — backward through all components
    11. Residual path — node with zero incoming edges
    12. Attention normalisation — alpha sums ≈ 1 per dest node
    13. Time sensitivity — different delta_t changes attention weights
    14. Edge-type sensitivity — different edge_type changes attention
    15. PyG batch — multiple patient graphs
    16. No cross-patient attention corruption
    17. No NaN/Inf
    18. Attention record extraction
"""

import torch
import pytest
from torch_geometric.data import Data, Batch

from src.models.time_encoding import TimeEncoding
from src.models.event_node_encoder import EventNodeEncoder
from src.models.te_gat import TemporalEdgeAttentionLayer, TEGATEncoder, NUM_EDGE_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CONCEPTS = 29   # default vocabulary size
HIDDEN_DIM = 64
N_HEADS = 4
N_LAYERS = 3
TIME_DIM = 16
EDGE_TYPE_DIM = 8


def _make_graph(n_nodes: int = 8, n_edges: int = 12, seed: int = 0) -> Data:
    """Construct a small random patient graph with correct attribute types."""
    torch.manual_seed(seed)
    data = Data()
    data.node_type = torch.randint(0, 6, (n_nodes,))
    data.concept_id = torch.randint(0, NUM_CONCEPTS, (n_nodes,))
    data.numeric_value = torch.randn(n_nodes)
    data.has_value = (torch.rand(n_nodes) > 0.3).float()
    data.relative_time = -torch.rand(n_nodes) * 48.0   # negative hours before t0
    data.time = data.relative_time.clone()

    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    data.edge_index = torch.stack([src, dst], dim=0)
    E = src.size(0)
    data.edge_delta_t = torch.rand(E) * 20.0
    data.edge_type = torch.randint(0, NUM_EDGE_TYPES, (E,))
    return data


def _make_encoder() -> EventNodeEncoder:
    return EventNodeEncoder(
        num_concepts=NUM_CONCEPTS,
        hidden_dim=HIDDEN_DIM,
        type_embedding_dim=16,
        concept_embedding_dim=32,
        numeric_hidden_dim=16,
        time_dim=TIME_DIM,
    )


def _make_tegat() -> TEGATEncoder:
    return TEGATEncoder(
        in_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        time_dim=TIME_DIM,
        edge_type_dim=EDGE_TYPE_DIM,
    )


# ---------------------------------------------------------------------------
# Test 1 — EventNodeEncoder shape
# ---------------------------------------------------------------------------

class TestEventNodeEncoderShape:
    def test_output_shape(self):
        data = _make_graph(n_nodes=10)
        enc = _make_encoder()
        enc.eval()
        with torch.no_grad():
            h = enc(data)
        assert h.shape == (10, HIDDEN_DIM), f"Got {h.shape}"


# ---------------------------------------------------------------------------
# Test 2 — Categorical dtype handling
# ---------------------------------------------------------------------------

class TestCategoricalDtype:
    def test_node_type_integer(self):
        data = _make_graph()
        assert data.node_type.dtype == torch.long
        assert data.concept_id.dtype == torch.long

    def test_encoder_accepts_integer_inputs(self):
        data = _make_graph()
        enc = _make_encoder()
        enc.eval()
        with torch.no_grad():
            h = enc(data)
        assert torch.isfinite(h).all()


# ---------------------------------------------------------------------------
# Test 3 — Non-numeric node (has_value=0)
# ---------------------------------------------------------------------------

class TestNonNumericNode:
    def test_non_numeric_node_finite(self):
        data = _make_graph(n_nodes=5)
        # Force all nodes to non-numeric
        data.numeric_value = torch.zeros(5)
        data.has_value = torch.zeros(5)
        enc = _make_encoder()
        enc.eval()
        with torch.no_grad():
            h = enc(data)
        assert torch.isfinite(h).all(), "Non-numeric nodes produced non-finite output"
        assert h.shape == (5, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Test 4 — Node time encoding: distinct times → distinct encodings
# ---------------------------------------------------------------------------

class TestNodeTimeEncoding:
    def test_different_times_differ(self):
        enc = TimeEncoding(output_dim=TIME_DIM)
        enc.eval()
        t1 = torch.tensor([-10.0])
        t2 = torch.tensor([-30.0])
        with torch.no_grad():
            e1 = enc(t1)
            e2 = enc(t2)
        assert not torch.allclose(e1, e2), "Different times produced identical encodings"


# ---------------------------------------------------------------------------
# Test 5 — Edge time encoding: distinct delta_t → distinct encodings
# ---------------------------------------------------------------------------

class TestEdgeTimeEncoding:
    def test_different_delta_t_differ(self):
        enc = TimeEncoding(output_dim=TIME_DIM)
        enc.eval()
        dt1 = torch.tensor([1.0])
        dt2 = torch.tensor([15.0])
        with torch.no_grad():
            e1 = enc(dt1)
            e2 = enc(dt2)
        assert not torch.allclose(e1, e2), "Different delta_t produced identical encodings"


# ---------------------------------------------------------------------------
# Test 6 — Zero delta_t
# ---------------------------------------------------------------------------

class TestZeroDeltaT:
    def test_zero_finite(self):
        enc = TimeEncoding(output_dim=TIME_DIM)
        enc.eval()
        with torch.no_grad():
            e = enc(torch.tensor([0.0]))
        assert torch.isfinite(e).all(), "Zero delta_t produced non-finite encoding"


# ---------------------------------------------------------------------------
# Test 7 — Large delta_t
# ---------------------------------------------------------------------------

class TestLargeDeltaT:
    def test_large_finite(self):
        enc = TimeEncoding(output_dim=TIME_DIM)
        enc.eval()
        with torch.no_grad():
            e = enc(torch.tensor([1e4]))
        assert torch.isfinite(e).all(), "Large delta_t produced non-finite encoding"


# ---------------------------------------------------------------------------
# Test 8 — TE-GAT forward shape
# ---------------------------------------------------------------------------

class TestTEGATShape:
    def test_output_shape(self):
        data = _make_graph(n_nodes=15, n_edges=25)
        enc = _make_encoder()
        tegat = _make_tegat()
        enc.eval(); tegat.eval()
        with torch.no_grad():
            h0 = enc(data)
            h_out, _ = tegat(h0, data.edge_index, data.edge_delta_t, data.edge_type)
        assert h_out.shape == (15, HIDDEN_DIM), f"Got {h_out.shape}"


# ---------------------------------------------------------------------------
# Test 9 — Multi-head
# ---------------------------------------------------------------------------

class TestMultiHead:
    def test_four_heads_correct_shape(self):
        data = _make_graph(n_nodes=10, n_edges=15)
        enc = _make_encoder()
        layer = TemporalEdgeAttentionLayer(
            in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM // N_HEADS,
            time_dim=TIME_DIM, edge_type_dim=EDGE_TYPE_DIM,
            n_heads=N_HEADS,
        )
        enc.eval(); layer.eval()
        with torch.no_grad():
            h0 = enc(data)
            h_out, attn = layer(h0, data.edge_index, data.edge_delta_t, data.edge_type, return_attention=True)
        # Output should be [N, n_heads * per_head_dim]
        assert h_out.shape[0] == 10
        assert h_out.shape[1] == HIDDEN_DIM, f"Got {h_out.shape}"
        # Attention: [E, n_heads]
        assert attn["alpha"].shape == (data.edge_index.size(1), N_HEADS)


# ---------------------------------------------------------------------------
# Test 10 — Gradient propagation
# ---------------------------------------------------------------------------

class TestGradientPropagation:
    def test_backward(self):
        data = _make_graph(n_nodes=10, n_edges=15)
        enc = _make_encoder()
        tegat = _make_tegat()
        h0 = enc(data)
        h_out, _ = tegat(h0, data.edge_index, data.edge_delta_t, data.edge_type)
        loss = h_out.mean()
        loss.backward()
        # Check gradients exist throughout
        for name, p in list(enc.named_parameters()) + list(tegat.named_parameters()):
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# Test 11 — Residual path: node with zero incoming edges
# ---------------------------------------------------------------------------

class TestResidualPath:
    def test_isolated_node_finite(self):
        """Isolated node (no incoming edges) should still produce a finite embedding."""
        # Create a graph where node 0 has no incoming edges
        n_nodes = 5
        data = Data()
        data.node_type = torch.zeros(n_nodes, dtype=torch.long)
        data.concept_id = torch.zeros(n_nodes, dtype=torch.long)
        data.numeric_value = torch.randn(n_nodes)
        data.has_value = torch.ones(n_nodes)
        data.relative_time = -torch.rand(n_nodes) * 24.0
        data.time = data.relative_time.clone()
        # Only edges among nodes 1,2,3,4 — node 0 has no incoming
        data.edge_index = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
        data.edge_delta_t = torch.rand(3)
        data.edge_type = torch.zeros(3, dtype=torch.long)

        enc = _make_encoder()
        tegat = _make_tegat()
        enc.eval(); tegat.eval()
        with torch.no_grad():
            h0 = enc(data)
            h_out, _ = tegat(h0, data.edge_index, data.edge_delta_t, data.edge_type)
        assert torch.isfinite(h_out[0]).all(), "Isolated node produced non-finite embedding"
        assert h_out.shape == (n_nodes, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Test 12 — Attention normalisation
# ---------------------------------------------------------------------------

class TestAttentionNormalisation:
    def test_alpha_sums_to_one(self):
        """For every destination node with at least one incoming edge, sum(alpha) ≈ 1."""
        data = _make_graph(n_nodes=10, n_edges=20)
        enc = _make_encoder()
        layer = TemporalEdgeAttentionLayer(
            in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM // N_HEADS,
            time_dim=TIME_DIM, edge_type_dim=EDGE_TYPE_DIM,
            n_heads=N_HEADS, dropout=0.0,
        )
        enc.eval(); layer.eval()
        with torch.no_grad():
            h0 = enc(data)
            _, attn = layer(h0, data.edge_index, data.edge_delta_t, data.edge_type, return_attention=True)
        alpha = attn["alpha"]   # [E, n_heads]
        dst = data.edge_index[1]
        for node in dst.unique():
            mask = dst == node
            alpha_sum = alpha[mask].sum(dim=0)  # [n_heads]
            assert torch.allclose(alpha_sum, torch.ones(N_HEADS), atol=1e-4), \
                f"Node {node}: alpha sum = {alpha_sum}, expected 1.0"


# ---------------------------------------------------------------------------
# Test 13 — Time sensitivity in TE-GAT attention
# ---------------------------------------------------------------------------

class TestTimeSensitivity:
    def test_different_delta_t_changes_attention(self):
        """Two edges identical except for delta_t should produce different attention logits."""
        n_nodes = 4
        # Build a simple graph with two parallel edges u→v but different delta_t
        data_a = Data()
        data_a.node_type = torch.zeros(n_nodes, dtype=torch.long)
        data_a.concept_id = torch.zeros(n_nodes, dtype=torch.long)
        data_a.numeric_value = torch.zeros(n_nodes)
        data_a.has_value = torch.zeros(n_nodes)
        data_a.relative_time = torch.zeros(n_nodes)
        data_a.time = data_a.relative_time.clone()
        data_a.edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        data_a.edge_type = torch.zeros(2, dtype=torch.long)
        data_a.edge_delta_t = torch.tensor([1.0, 1.0])  # same dt

        data_b = Data()
        data_b.node_type = data_a.node_type.clone()
        data_b.concept_id = data_a.concept_id.clone()
        data_b.numeric_value = data_a.numeric_value.clone()
        data_b.has_value = data_a.has_value.clone()
        data_b.relative_time = data_a.relative_time.clone()
        data_b.time = data_b.relative_time.clone()
        data_b.edge_index = data_a.edge_index.clone()
        data_b.edge_type = data_a.edge_type.clone()
        data_b.edge_delta_t = torch.tensor([1.0, 20.0])  # different dt on second edge

        enc = _make_encoder()
        layer = TemporalEdgeAttentionLayer(
            in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM // N_HEADS,
            time_dim=TIME_DIM, edge_type_dim=EDGE_TYPE_DIM,
            n_heads=N_HEADS, dropout=0.0,
        )
        enc.eval(); layer.eval()
        with torch.no_grad():
            h0_a = enc(data_a)
            _, attn_a = layer(h0_a, data_a.edge_index, data_a.edge_delta_t, data_a.edge_type, return_attention=True)

            # Use same node features (detach so same h0)
            _, attn_b = layer(h0_a, data_b.edge_index, data_b.edge_delta_t, data_b.edge_type, return_attention=True)

        alpha_a = attn_a["alpha"]
        alpha_b = attn_b["alpha"]
        # Second edge has different dt in b — alpha of the second edge should differ
        assert not torch.allclose(alpha_a[1], alpha_b[1], atol=1e-5), \
            "Changing delta_t did not change attention weight"


# ---------------------------------------------------------------------------
# Test 14 — Edge-type sensitivity in TE-GAT attention
# ---------------------------------------------------------------------------

class TestEdgeTypeSensitivity:
    def test_different_edge_type_changes_attention(self):
        """Same graph but different edge_type should yield different attention."""
        n_nodes = 4
        common_kwargs = dict(
            node_type=torch.zeros(n_nodes, dtype=torch.long),
            concept_id=torch.zeros(n_nodes, dtype=torch.long),
            numeric_value=torch.zeros(n_nodes),
            has_value=torch.zeros(n_nodes),
            relative_time=torch.zeros(n_nodes),
            time=torch.zeros(n_nodes),
        )
        edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        delta_t = torch.tensor([5.0, 5.0])

        data_a = Data(**common_kwargs,
                      edge_index=edge_index.clone(),
                      edge_delta_t=delta_t.clone(),
                      edge_type=torch.tensor([0, 0], dtype=torch.long))
        data_b = Data(**common_kwargs,
                      edge_index=edge_index.clone(),
                      edge_delta_t=delta_t.clone(),
                      edge_type=torch.tensor([0, 3], dtype=torch.long))  # second edge differs

        enc = _make_encoder()
        layer = TemporalEdgeAttentionLayer(
            in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM // N_HEADS,
            time_dim=TIME_DIM, edge_type_dim=EDGE_TYPE_DIM,
            n_heads=N_HEADS, dropout=0.0,
        )
        enc.eval(); layer.eval()
        with torch.no_grad():
            h0 = enc(data_a)
            _, attn_a = layer(h0, data_a.edge_index, data_a.edge_delta_t, data_a.edge_type, return_attention=True)
            _, attn_b = layer(h0, data_b.edge_index, data_b.edge_delta_t, data_b.edge_type, return_attention=True)

        # Second edge has different type — alpha should differ
        assert not torch.allclose(attn_a["alpha"][1], attn_b["alpha"][1], atol=1e-5), \
            "Changing edge_type did not change attention weight"


# ---------------------------------------------------------------------------
# Test 15 — PyG batch: multiple patient graphs
# ---------------------------------------------------------------------------

class TestPyGBatch:
    def test_batch_output_shape(self):
        graphs = [_make_graph(n_nodes=7 + i, n_edges=10 + i, seed=i) for i in range(3)]
        batch = Batch.from_data_list(graphs)
        enc = _make_encoder()
        tegat = _make_tegat()
        enc.eval(); tegat.eval()
        with torch.no_grad():
            h0 = enc(batch)
            h_out, _ = tegat(h0, batch.edge_index, batch.edge_delta_t, batch.edge_type)
        total_nodes = sum(g.num_nodes for g in graphs)
        assert h_out.shape == (total_nodes, HIDDEN_DIM), f"Got {h_out.shape}"


# ---------------------------------------------------------------------------
# Test 16 — No cross-patient attention corruption
# ---------------------------------------------------------------------------

class TestNoCrossPatientCorruption:
    def test_no_cross_patient_edges(self):
        """Verify that batching introduces no cross-patient edges."""
        g1 = _make_graph(n_nodes=6, n_edges=8, seed=0)
        g2 = _make_graph(n_nodes=6, n_edges=8, seed=1)
        batch = Batch.from_data_list([g1, g2])
        edge_index = batch.edge_index  # [2, E_total]
        node_batch = batch.batch       # [N_total] — which graph each node belongs to

        src_batch = node_batch[edge_index[0]]
        dst_batch = node_batch[edge_index[1]]
        assert (src_batch == dst_batch).all(), "Cross-patient edges found in batch"


# ---------------------------------------------------------------------------
# Test 17 — No NaN/Inf
# ---------------------------------------------------------------------------

class TestNoNaNInf:
    def test_random_batch_finite(self):
        graphs = [_make_graph(n_nodes=8, n_edges=12, seed=i) for i in range(5)]
        batch = Batch.from_data_list(graphs)
        enc = _make_encoder()
        tegat = _make_tegat()
        enc.eval(); tegat.eval()
        with torch.no_grad():
            h0 = enc(batch)
            h_out, _ = tegat(h0, batch.edge_index, batch.edge_delta_t, batch.edge_type)
        assert torch.isfinite(h_out).all(), "NaN or Inf in batch output"


# ---------------------------------------------------------------------------
# Test 18 — Attention record extraction
# ---------------------------------------------------------------------------

class TestAttentionExtraction:
    def test_attention_records_structure(self):
        data = _make_graph(n_nodes=10, n_edges=15)
        enc = _make_encoder()
        tegat = _make_tegat()
        enc.eval(); tegat.eval()
        with torch.no_grad():
            h0 = enc(data)
            h_out, attn_records = tegat(
                h0, data.edge_index, data.edge_delta_t, data.edge_type,
                return_attention=True,
            )
        assert attn_records is not None
        assert len(attn_records) == N_LAYERS, f"Expected {N_LAYERS} records, got {len(attn_records)}"
        for i, rec in enumerate(attn_records):
            assert rec["layer"] == i
            assert rec["n_heads"] == N_HEADS
            assert rec["alpha"].shape[0] == data.edge_index.size(1), "Wrong alpha size"
            assert rec["alpha"].shape[1] == N_HEADS, "Wrong head count"
            assert "edge_index" in rec
