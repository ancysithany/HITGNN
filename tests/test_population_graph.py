"""
Tests for Inductive Population Graph (Phase 8).
"""
import pytest
import torch
import numpy as np
from src.models.population_graph import PopulationGraphModule

@pytest.fixture
def pop_config():
    return {
        "population_graph": {
            "k_neighbors": 2,
            "lambda_embed": 0.5,
            "lambda_meta": 0.5
        },
        "pooling": {
            "fused_dim": 4
        }
    }

def test_meta_normalizer_train_only():
    """1. meta normalizer train-only"""
    from src.preprocessing.meta_normalizer import PatientMetaNormalizer
    from src.data.synthetic_generator import PatientRecord
    
    p1 = PatientRecord("p1", events=[], mortality=0, age=50.0, sex="M", icu_type="MICU", sofa_score=2.0, comorbidities={}, t0=0.0)
    p2 = PatientRecord("p2", events=[], mortality=0, age=70.0, sex="F", icu_type="SICU", sofa_score=6.0, comorbidities={}, t0=0.0)
    
    norm = PatientMetaNormalizer()
    norm.fit([p1, p2])
    
    # 50, 70 -> mean 60, std 10
    # 2, 6 -> mean 4, std 2
    assert norm.age_mean == 60
    assert norm.age_std == 10
    assert norm.sofa_mean == 4
    assert norm.sofa_std == 2
    
    age_n, sofa_n = norm.transform(70, 6)
    assert age_n == 1.0
    assert sofa_n == 1.0

def test_similarity_scalar_correctness(pop_config):
    """2. similarity scalar correctness"""
    mod = PopulationGraphModule(pop_config)
    
    q_emb = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_meta = torch.zeros((1, 10))
    
    r_emb = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    r_meta = torch.zeros((2, 10))
    
    sim = mod._compute_similarity(q_emb, q_meta, r_emb, r_meta)
    
    assert sim.shape == (1, 2)
    # Cosine(1) * 0.5 + exp(0) * 0.5 = 1.0
    assert torch.isclose(sim[0, 0], torch.tensor(1.0))
    # Cosine(0) * 0.5 + exp(0) * 0.5 = 0.5
    assert torch.isclose(sim[0, 1], torch.tensor(0.5))

def test_cosine_finite_with_zero_vector(pop_config):
    """3. cosine finite with zero vector"""
    mod = PopulationGraphModule(pop_config)
    q_emb = torch.zeros((1, 4))
    r_emb = torch.ones((1, 4))
    q_meta = torch.zeros((1, 10))
    r_meta = torch.zeros((1, 10))
    
    sim = mod._compute_similarity(q_emb, q_meta, r_emb, r_meta)
    assert torch.isfinite(sim).all()

def test_top_k_shape_and_r_lt_k(pop_config):
    """4. top-k shape & 5. R < k handling"""
    mod = PopulationGraphModule(pop_config) # k = 2
    
    # 3 Queries, 1 Reference
    q_emb = torch.randn(3, 4)
    q_meta = torch.zeros((3, 10))
    
    r_emb = torch.randn(1, 4)
    r_meta = torch.zeros((1, 10))
    
    # Forward
    h_prime, topk_idx, topk_sim, alpha = mod(q_emb, q_meta, r_emb, r_meta)
    
    assert h_prime.shape == (3, 4)
    assert topk_idx.shape == (3, 1) # R < k
    assert topk_sim.shape == (3, 1)
    assert alpha.shape == (3, 1)
    
def test_train_self_exclusion(pop_config):
    """6. train self-exclusion"""
    mod = PopulationGraphModule(pop_config) # k = 2
    
    q_emb = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    q_meta = torch.zeros((2, 10))
    q_ids = ["P1", "P2"]
    
    # references are identical to queries
    r_emb = q_emb.clone()
    r_meta = q_meta.clone()
    r_ids = ["P1", "P2"]
    
    # Forward
    h_prime, topk_idx, topk_sim, alpha = mod(q_emb, q_meta, r_emb, r_meta, q_ids, r_ids)
    
    # P1 should select P2, not P1
    assert topk_idx[0, 0].item() == 1
    # P2 should select P1, not P2
    assert topk_idx[1, 0].item() == 0

def test_attention_sums_to_1(pop_config):
    """14. attention sums to 1"""
    mod = PopulationGraphModule(pop_config)
    
    q_emb = torch.randn(2, 4)
    q_meta = torch.zeros((2, 10))
    
    r_emb = torch.randn(5, 4)
    r_meta = torch.zeros((5, 10))
    
    h_prime, topk_idx, topk_sim, alpha = mod(q_emb, q_meta, r_emb, r_meta)
    
    # alpha should sum to 1 over k (k=2)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2))
    
def test_neighbor_intervention_changes_query(pop_config):
    """21. neighbor intervention changes query"""
    mod = PopulationGraphModule(pop_config)
    q_emb = torch.randn(1, 4)
    q_meta = torch.zeros((1, 10))
    
    r_emb = torch.randn(2, 4)
    r_meta = torch.zeros((2, 10))
    
    h_prime_1, _, _, _ = mod(q_emb, q_meta, r_emb, r_meta)
    
    # intervene on neighbor 0
    r_emb_new = r_emb.clone()
    r_emb_new[0] += 5.0
    
    h_prime_2, _, _, _ = mod(q_emb, q_meta, r_emb_new, r_meta)
    
    assert not torch.allclose(h_prime_1, h_prime_2)

def test_non_neighbor_intervention_does_not_change_query(pop_config):
    """22. non-neighbor intervention does not change query"""
    mod = PopulationGraphModule(pop_config) # k=2
    q_emb = torch.randn(1, 4)
    q_meta = torch.zeros((1, 10))
    
    r_emb = torch.randn(5, 4)
    r_meta = torch.zeros((5, 10))
    
    # Make sure reference 4 is terrible (opposite of query)
    r_emb[4] = -q_emb[0]
    # Make metadata distance large
    r_meta[4] = torch.ones(10) * 100.0
    
    h_prime_1, topk_idx, _, _ = mod(q_emb, q_meta, r_emb, r_meta)
    assert 4 not in topk_idx[0] # it's not a neighbor
    
    # intervene on non-neighbor 4
    r_emb_new = r_emb.clone()
    r_emb_new[4] = -q_emb[0] * 2.0 # still terrible
    
    h_prime_2, _, _, _ = mod(q_emb, q_meta, r_emb_new, r_meta)
    
    assert torch.allclose(h_prime_1, h_prime_2)

def test_query_query_isolation(pop_config):
    """23. query-query isolation"""
    mod = PopulationGraphModule(pop_config)
    
    q_emb = torch.randn(2, 4)
    q_meta = torch.zeros((2, 10))
    
    r_emb = torch.randn(3, 4)
    r_meta = torch.zeros((3, 10))
    
    # Batch of 2
    h_prime_batch, _, _, _ = mod(q_emb, q_meta, r_emb, r_meta)
    
    # Run separately
    h_prime_q0, _, _, _ = mod(q_emb[0:1], q_meta[0:1], r_emb, r_meta)
    h_prime_q1, _, _, _ = mod(q_emb[1:2], q_meta[1:2], r_emb, r_meta)
    
    assert torch.allclose(h_prime_batch[0], h_prime_q0.squeeze())
    assert torch.allclose(h_prime_batch[1], h_prime_q1.squeeze())

def test_gradient_flow(pop_config):
    """17, 18. gradient flow"""
    mod = PopulationGraphModule(pop_config)
    q_emb = torch.randn(1, 4, requires_grad=True)
    q_meta = torch.zeros((1, 10))
    r_emb = torch.randn(2, 4, requires_grad=True)
    r_meta = torch.zeros((2, 10))
    
    h_prime, _, _, _ = mod(q_emb, q_meta, r_emb, r_meta)
    loss = h_prime.sum()
    loss.backward()
    
    assert q_emb.grad is not None
    assert r_emb.grad is not None
    assert mod.W_neighbor.weight.grad is not None
