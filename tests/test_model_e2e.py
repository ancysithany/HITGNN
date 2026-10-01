"""
Test full HITGNN Phase 7 model end-to-end.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.models.hitgnn import PatientMortalityModel
from src.training.trainer import Trainer
from src.data.synthetic_generator import SyntheticCohortGenerator
from src.data.vocab import ConceptVocabulary
from tests.test_hierarchical_pool import _make_batch


def test_prediction_head_output_shape():
    """TEST 1 & 2: PredictionHead output shape and returns raw logits."""
    config = {
        "te_gat": {"hidden_dim": 64, "n_heads": 4, "n_layers": 3, "time_encoding_dim": 16, "edge_type_dim": 8},
        "missingness": {"hidden_dim": 64},
        "pooling": {"fused_dim": 128, "dropout": 0.0},
        "prediction_head": {"hidden_dim": 64, "dropout": 0.0}
    }
    model = PatientMortalityModel(config, vocab_size=29)
    batch = _make_batch(num_graphs=4)
    
    logits = model(batch)
    
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    # Logits shouldn't be bounded in [0, 1] purely
    
    
def test_sigmoid_probability():
    """TEST 3: sigmoid(logit) is valid probability."""
    logits = torch.tensor([-10.0, 0.0, 2.5, 10.0])
    probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_bce_with_logits():
    """TEST 4: BCEWithLogitsLoss works."""
    loss_fn = nn.BCEWithLogitsLoss()
    logits = torch.tensor([0.0, 2.0])
    y = torch.tensor([0.0, 1.0])
    loss = loss_fn(logits, y)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_forward_backward_optimizer():
    """TEST 6, 7, 8, 9: forward pass, backward pass, finite gradients, optimizer update."""
    config = {
        "te_gat": {"hidden_dim": 64, "n_heads": 4, "n_layers": 3, "time_encoding_dim": 16, "edge_type_dim": 8},
        "missingness": {"hidden_dim": 64},
        "pooling": {"fused_dim": 128, "dropout": 0.0},
        "prediction_head": {"hidden_dim": 64, "dropout": 0.0}
    }
    model = PatientMortalityModel(config, vocab_size=29)
    
    # Save original weights to check update
    orig_weight = model.prediction_head.mlp[0].weight.clone()
    
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    batch = _make_batch(num_graphs=4)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    opt.zero_grad()
    logits = model(batch)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    
    # Check finite gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
            
    opt.step()
    
    new_weight = model.prediction_head.mlp[0].weight
    assert not torch.allclose(orig_weight, new_weight)


def test_label_isolation():
    """TEST 14, 23 (W): label isolation."""
    config = {
        "te_gat": {"hidden_dim": 64, "n_heads": 4, "n_layers": 3, "time_encoding_dim": 16, "edge_type_dim": 8},
        "missingness": {"hidden_dim": 64},
        "pooling": {"fused_dim": 128, "dropout": 0.0},
        "prediction_head": {"hidden_dim": 64, "dropout": 0.0}
    }
    model = PatientMortalityModel(config, vocab_size=29)
    model.eval()
    
    batch = _make_batch(num_graphs=2)
    batch.y = torch.tensor([0.0, 0.0])
    
    with torch.no_grad():
        logits_0 = model(batch)
        
        batch.y = torch.tensor([1.0, 1.0])
        logits_1 = model(batch)
        
    assert torch.allclose(logits_0, logits_1)


def test_patient_isolation():
    """TEST 15, 22 (V): patient isolation."""
    config = {
        "te_gat": {"hidden_dim": 64, "n_heads": 4, "n_layers": 3, "time_encoding_dim": 16, "edge_type_dim": 8},
        "missingness": {"hidden_dim": 64},
        "pooling": {"fused_dim": 128, "dropout": 0.0},
        "prediction_head": {"hidden_dim": 64, "dropout": 0.0}
    }
    model = PatientMortalityModel(config, vocab_size=29)
    model.eval()
    
    batch = _make_batch(num_graphs=2)
    
    with torch.no_grad():
        logits_base = model(batch)
        
        batch_mod = batch.clone()
        batch_mod.numeric_value = batch.numeric_value.clone()
        idx_p0 = (batch.batch == 0)
        batch_mod.numeric_value[idx_p0] += 5.0
        
        logits_mod = model(batch_mod)
        
    assert not torch.allclose(logits_base[0], logits_mod[0])
    assert torch.allclose(logits_base[1], logits_mod[1])
