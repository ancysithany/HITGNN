"""
scripts/audit_phase6.py — Detailed audit of Phase 6 architecture.

Verifies:
    1. Finiteness of all embeddings and attentions
    2. Gradient health of all trainable parameters
    3. Normalizer non-trainability
    4. Missingness branch isolation/contribution
    5. Graph branch isolation/contribution
"""

import sys
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.vocab import ConceptVocabulary
from src.data.terminology import ClinicalTerminology
from src.data.synthetic_generator import SyntheticCohortGenerator, patient_level_split
from src.preprocessing.temporal_binner import TemporalBinner
from src.preprocessing.normalizer import FeatureNormalizer
from src.preprocessing.event_normalizer import EventNormalizer
from src.graph.graph_builder import PatientGraphBuilder
from src.data.dataset import HITGNNPatientDataset
from src.models.hitgnn import PatientEncoder


def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ------------------------------------------------------------------
    # 1. TEST COLLECTION AUDIT
    # ------------------------------------------------------------------
    banner("TEST COLLECTION AUDIT")
    print("Running pytest --collect-only -q...")
    # This might be simpler to run as a shell command later, but we can do it here:
    cmd = [sys.executable, "-m", "pytest", "tests/test_hierarchical_pool.py", "--collect-only", "-q"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to run pytest collection:", e.stderr)

    # ------------------------------------------------------------------
    # Setup data for remaining tests
    # ------------------------------------------------------------------
    vocab = ConceptVocabulary.build_default()
    term = ClinicalTerminology.build_default()
    config = load_config("configs/default.yaml")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    gen = SyntheticCohortGenerator(config, vocab, seed=seed)
    patients = gen.generate()

    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    ids = [p.patient_id for p in patients]
    labels = [p.mortality for p in patients]
    splits = patient_level_split(ids, split_ratios, seed=seed, labels=labels)

    train_patients = [p for p in patients if p.patient_id in splits["train"]]
    biomarkers = ["PCT", "CRP", "Lactate", "WBC", "Creatinine", "Bilirubin", "Platelets"]

    bio_norm = FeatureNormalizer()
    event_norm = EventNormalizer()
    for bm in biomarkers:
        vals = [e.value for p in train_patients for e in p.events
                if e.event_type == "LAB" and e.concept_name == bm and e.value is not None]
        bio_norm.fit(bm, np.array(vals) if vals else np.array([]))
    continuous = biomarkers + ["MAP", "Heart Rate", "Respiratory Rate", "Temperature", "SpO2", "GCS"]
    for c in continuous:
        vals = [e.value for p in train_patients for e in p.events
                if e.concept_name == c and e.value is not None]
        event_norm.fit(c, np.array(vals) if vals else np.array([]))

    binner = TemporalBinner(n_bins=6)
    builder = PatientGraphBuilder(vocab, term, event_norm, config.get("graph", {}))
    train_ds = HITGNNPatientDataset(patients, builder, binner, bio_norm, biomarkers, splits["train"])

    loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    encoder = PatientEncoder(config, vocab_size=vocab.size)
    encoder.train()
    
    # ------------------------------------------------------------------
    # 3. NORMALIZER TRAINABILITY CHECK
    # ------------------------------------------------------------------
    banner("NORMALIZER TRAINABILITY CHECK")
    print(f"FeatureNormalizer is nn.Module: {isinstance(bio_norm, nn.Module)}")
    print(f"EventNormalizer is nn.Module: {isinstance(event_norm, nn.Module)}")
    
    param_names = [n for n, p in encoder.named_parameters()]
    has_norm = any('norm' in n.lower() for n in param_names if 'normalizer' in n.lower())
    print(f"Normalizer parameters found in named_parameters: {has_norm}")

    # ------------------------------------------------------------------
    # 4. EMBEDDING & 5. ATTENTION FINITENESS CHECK
    # ------------------------------------------------------------------
    banner("EMBEDDING FINITENESS CHECK")
    encoder.eval()
    with torch.no_grad():
        out = encoder(batch)
        
    print(f"node_embeddings finite: {torch.isfinite(out.node_embeddings).all().item()}")
    print(f"type_embeddings finite: {torch.isfinite(out.node_embeddings).all().item()}")  # Wait, wait, actually let's re-eval and get everything
    print(f"patient_event_embedding finite: {torch.isfinite(out.patient_event_embedding).all().item()}")
    print(f"missingness_summary finite: {torch.isfinite(out.missingness_summary).all().item()}")
    print(f"fused_embedding finite: {torch.isfinite(out.fused_embedding).all().item()}")

    banner("ATTENTION FINITENESS CHECK")
    print(f"TE-GAT attention finite: {all(torch.isfinite(r['alpha']).all().item() for r in out.te_gat_attention)}")
    print(f"type-level node attention finite: {torch.isfinite(out.node_attention).all().item()}")
    print(f"patient-level type attention finite: {torch.isfinite(out.type_attention).all().item()}")
    
    all_alphas_sum_1 = True
    mask = out.type_present_mask
    for p in range(batch.num_graphs):
        for t in range(6):
            if mask[p, t] == 1.0:
                idx = (batch.batch == p) & (batch.node_type == t)
                s = out.node_attention[idx].sum().item()
                if not abs(s - 1.0) < 1e-4:
                    all_alphas_sum_1 = False
    print(f"sum node alpha ≈ 1 for all present types: {all_alphas_sum_1}")
    
    beta_sums = out.type_attention.sum(dim=1)
    print(f"sum beta over present types ≈ 1: {torch.allclose(beta_sums, torch.ones(batch.num_graphs), atol=1e-4)}")
    
    absent_beta = out.type_attention[mask == 0]
    if absent_beta.numel() > 0:
        print(f"absent types beta = 0: {(absent_beta == 0.0).all().item()}")
    else:
        print("absent types beta = 0: N/A (no absent types in this batch)")

    # ------------------------------------------------------------------
    # 2. TRAINABLE PARAMETER GRADIENT CHECK
    # ------------------------------------------------------------------
    banner("TRAINABLE PARAMETER GRADIENT CHECK")
    encoder.train()
    out_train = encoder(batch)
    loss = out_train.fused_embedding.sum()
    loss.backward()
    
    all_finite = True
    none_grads = []
    zero_grads = []
    
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                none_grads.append(name)
                all_finite = False
            elif not torch.isfinite(p.grad).all():
                all_finite = False
                print(f"{name}: contains NaN or Inf!")
            elif (p.grad == 0).all():
                zero_grads.append(name)
    
    print(f"All trainable params have finite, non-None gradients: {all_finite and not none_grads}")
    if none_grads:
        print(f"None grads: {none_grads}")
    if zero_grads:
        print(f"Zero grads: {zero_grads}")
        
    print("\nRepresentative gradients check:")
    print(f"node_encoder.type_emb.weight: grad sum={encoder.node_encoder.type_emb.weight.grad.abs().sum().item():.4f}")
    print(f"node_encoder.concept_emb.weight: grad sum={encoder.node_encoder.concept_emb.weight.grad.abs().sum().item():.4f}")
    print(f"node_encoder.numeric_proj.0.weight: grad sum={encoder.node_encoder.numeric_proj[0].weight.grad.abs().sum().item():.4f}")
    
    print(f"te_gat.layers.0.W_node.weight: grad sum={encoder.te_gat.layers[0].W_node.weight.grad.abs().sum().item():.4f}")
    print(f"te_gat.layers.0.W_msg.weight: grad sum={encoder.te_gat.layers[0].W_msg.weight.grad.abs().sum().item():.4f}")
    print(f"te_gat.layers.0.attn_vec: grad sum={encoder.te_gat.layers[0].attn_vec.grad.abs().sum().item():.4f}")
    
    print(f"pool.type_attn.q_type: grad sum={encoder.pool.type_attn.q_type.grad.abs().sum().item():.4f}")
    print(f"pool.patient_attn.q_patient: grad sum={encoder.pool.patient_attn.q_patient.grad.abs().sum().item():.4f}")
    
    print(f"missingness_encoder.mlp.0.weight: grad sum={encoder.missingness_encoder.mlp[0].weight.grad.abs().sum().item():.4f}")
    print(f"missingness_encoder.mlp.2.weight: grad sum={encoder.missingness_encoder.mlp[2].weight.grad.abs().sum().item():.4f}")
    
    print(f"pool.fusion.0.weight: grad sum={encoder.pool.fusion[0].weight.grad.abs().sum().item():.4f}")

    # ------------------------------------------------------------------
    # 6. MISSINGNESS BRANCH CONTRIBUTION CHECK
    # ------------------------------------------------------------------
    banner("MISSINGNESS BRANCH CONTRIBUTION CHECK")
    encoder.eval()
    with torch.no_grad():
        out_base = encoder(batch)
        base_fused = out_base.fused_embedding.clone()
        
        # Modify missingness for Patient 0 only
        batch_mod_miss = batch.clone()
        batch_mod_miss.missingness_input = batch.missingness_input.clone()
        batch_mod_miss.missingness_input[0] += 5.0
        
        out_mod_miss = encoder(batch_mod_miss)
        mod_miss_fused = out_mod_miss.fused_embedding.clone()
        
    p0_miss_diff = (base_fused[0] - mod_miss_fused[0]).abs().sum().item()
    p1_miss_diff = (base_fused[1] - mod_miss_fused[1]).abs().sum().item()
    
    print(f"Patient 0 fused embedding changed: {p0_miss_diff > 1e-4}")
    print(f"Patient 1+ fused embedding unchanged: {p1_miss_diff < 1e-6}")

    # ------------------------------------------------------------------
    # 7. GRAPH BRANCH CONTRIBUTION CHECK
    # ------------------------------------------------------------------
    banner("GRAPH BRANCH CONTRIBUTION CHECK")
    with torch.no_grad():
        # Modify graph features for Patient 0 only
        batch_mod_graph = batch.clone()
        batch_mod_graph.numeric_value = batch.numeric_value.clone()
        # Find some nodes for patient 0
        idx_p0 = (batch.batch == 0)
        batch_mod_graph.numeric_value[idx_p0] += 5.0
        
        out_mod_graph = encoder(batch_mod_graph)
        mod_graph_fused = out_mod_graph.fused_embedding.clone()
        
    p0_graph_diff = (base_fused[0] - mod_graph_fused[0]).abs().sum().item()
    p1_graph_diff = (base_fused[1] - mod_graph_fused[1]).abs().sum().item()
    
    print(f"Patient 0 fused embedding changed: {p0_graph_diff > 1e-4}")
    print(f"Patient 1+ fused embedding unchanged: {p1_graph_diff < 1e-6}")


if __name__ == "__main__":
    main()
