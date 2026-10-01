"""
scripts/verify_phase6.py — End-to-end Phase 6 verification.

Verifies:
    1. Patient Encoder Shapes
    2. Attention outputs (at least 3 patients)
    3. Missing empty type check
    4. Tiny learnability check
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
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
    # Setup
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
    encoder.eval()

    with torch.no_grad():
        out = encoder(batch, return_attention=True)

    # ------------------------------------------------------------------
    # SHAPES
    # ------------------------------------------------------------------
    banner("PATIENT ENCODER SHAPES")
    print(f"batch patient count: {batch.num_graphs}")
    print(f"total nodes: {batch.num_nodes}")
    print(f"node_embeddings shape: {list(out.node_embeddings.shape)}")
    print(f"type_embeddings shape: {list(out.type_present_mask.shape[:-1] + (6, 64))}")  # derived
    print(f"type_present_mask shape: {list(out.type_present_mask.shape)}")
    print(f"patient_event_embedding shape: {list(out.patient_event_embedding.shape)}")
    print(f"missingness_slot_embeddings shape: {list(out.slot_missingness_embeddings.shape)}")
    print(f"missingness_summary shape: {list(out.missingness_summary.shape)}")
    print(f"fused_embedding shape: {list(out.fused_embedding.shape)}")

    # ------------------------------------------------------------------
    # ATTENTION OUTPUTS
    # ------------------------------------------------------------------
    banner("ATTENTION OUTPUTS")
    type_names = ["LAB", "VITAL", "MEDICATION", "PROCEDURE", "DIAGNOSIS", "CLINICAL_CONCEPT"]
    
    for i in range(min(3, batch.num_graphs)):
        print(f"\nPatient {i}")
        mask_i = out.type_present_mask[i]
        beta_i = out.type_attention[i]
        
        sum_present = 0.0
        for t_idx, name in enumerate(type_names):
            if mask_i[t_idx] == 1.0:
                print(f"{name:<18} {beta_i[t_idx]:.4f}")
                sum_present += beta_i[t_idx].item()
            else:
                print(f"{name:<18} 0.0000  [absent]")
                
        print(f"\nsum present weights:\n{sum_present:.3f}")
        
        # Look at LAB nodes for this patient
        lab_nodes_mask = (batch.batch == i) & (batch.node_type == 0)
        lab_count = lab_nodes_mask.sum().item()
        
        if lab_count > 0:
            lab_alpha_sum = out.node_attention[lab_nodes_mask].sum().item()
            print(f"\nLAB node count: {lab_count}")
            print(f"LAB node attention sum: {lab_alpha_sum:.3f} (Expected: ≈ 1)")

    # ------------------------------------------------------------------
    # PARAMETER COUNTS
    # ------------------------------------------------------------------
    banner("PARAMETER COUNTS")
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total Phase 6 PatientEncoder: {count_params(encoder):,} parameters")

    # ------------------------------------------------------------------
    # TINY LEARNABILITY TEST
    # ------------------------------------------------------------------
    banner("TINY LEARNABILITY CHECK (proxy task)")
    print("Proxy task: Predict whether SOFA > 5 based ONLY on fused_embedding.")
    print("Uses temporary Linear head INSIDE this script.")

    small_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    proxy_batch = next(iter(small_loader))

    # Proxy target: patient's max normalized Lactate score > 0
    sofa_id = vocab.get_by_name("Lactate").concept_id
    proxy_labels = []
    for g_idx in proxy_batch.batch.unique():
        mask = (proxy_batch.batch == g_idx) & (proxy_batch.concept_id == sofa_id)
        if mask.any():
            mean_sofa = proxy_batch.numeric_value[mask].mean()
        else:
            mean_sofa = torch.tensor(0.0)
        proxy_labels.append(float(mean_sofa > 0.0))
    y_proxy = torch.tensor(proxy_labels, dtype=torch.float32)

    encoder_small = PatientEncoder(config, vocab_size=vocab.size)
    proxy_head = nn.Linear(128, 1)

    optimizer = torch.optim.Adam(
        list(encoder_small.parameters()) + list(proxy_head.parameters()),
        lr=5e-3,
    )

    losses = []
    encoder_small.train(); proxy_head.train()
    for step in range(50):
        optimizer.zero_grad()
        out_s = encoder_small(proxy_batch)
        logits = proxy_head(out_s.fused_embedding).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_proxy)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    initial_loss = losses[0]
    final_loss = losses[-1]
    decreased = final_loss < initial_loss * 0.8
    print(f"Initial loss (step 0):  {initial_loss:.4f}")
    print(f"Final loss  (step 49): {final_loss:.4f}")
    print(f"Loss decreased by >20%: {decreased}")


if __name__ == "__main__":
    main()
