"""
Phase 4 verification script — Temporal Event Graph Construction.

Demonstrates the full pipeline:
    1. Load synthetic cohort + splits
    2. Fit normalizers on TRAIN ONLY
    3. Build dataset and batch graphs
    4. Print statistics and examples
    5. Generate visualization
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.utils.config import load_config
from src.data.vocab import ConceptVocabulary
from src.data.terminology import ClinicalTerminology
from src.data.synthetic_generator import SyntheticCohortGenerator
from src.preprocessing.temporal_binner import TemporalBinner
from src.preprocessing.normalizer import FeatureNormalizer
from src.preprocessing.event_normalizer import EventNormalizer
from src.graph.graph_builder import PatientGraphBuilder, validate_patient_graph
from src.data.dataset import HITGNNPatientDataset
from src.visualization.graph import visualize_patient_graph


def main():
    parser = argparse.ArgumentParser(description="Phase 4 verification")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)

    # 1. Load data
    vocab = ConceptVocabulary.build_default()
    term = ClinicalTerminology.build_default()
    gen = SyntheticCohortGenerator(config, vocab, seed=seed)
    patients = gen.generate()

    # Reuse splits
    from src.data.synthetic_generator import patient_level_split
    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    ids = [p.patient_id for p in patients]
    labels = [p.mortality for p in patients]
    splits = patient_level_split(ids, split_ratios, seed=seed, labels=labels)
    
    train_patients = [p for p in patients if p.patient_id in splits["train"]]

    # 2. Fit normalizers
    print("Fitting normalizers on TRAIN...")
    biomarkers = ["PCT", "CRP", "Lactate", "WBC", "Creatinine", "Bilirubin", "Platelets"]
    bio_norm = FeatureNormalizer()
    event_norm = EventNormalizer()

    for bm in biomarkers:
        vals = [e.value for p in train_patients for e in p.events if e.event_type == "LAB" and e.concept_name == bm and e.value is not None]
        bio_norm.fit(bm, np.array(vals) if vals else np.array([]))

    # All continuous for graph
    continuous_concepts = biomarkers + ["MAP", "Heart Rate", "Respiratory Rate", "Temperature", "SpO2", "GCS"]
    for conc in continuous_concepts:
        vals = [e.value for p in train_patients for e in p.events if e.concept_name == conc and e.value is not None]
        event_norm.fit(conc, np.array(vals) if vals else np.array([]))

    os.makedirs("data/processed", exist_ok=True)
    event_norm.save("data/processed/event_normalizer.json")

    # 3. Build Datasets
    print("Building datasets...")
    binner = TemporalBinner(n_bins=6)
    builder = PatientGraphBuilder(vocab, term, event_norm, config.get("graph", {}))

    train_ds = HITGNNPatientDataset(patients, builder, binner, bio_norm, biomarkers, splits["train"])
    
    # 4. Batching
    print("Verifying batching...")
    loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    print(f"Batch metadata shape: {batch.patient_meta.shape}")
    print(f"Batch missingness shape: {batch.missingness_input.shape}")
    print(f"Batch nodes: {batch.num_nodes}, Batch edges: {batch.num_edges}")

    # 5. Stats and Validation
    nodes = []
    edges = []
    edge_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print("\nValidating all train graphs...")
    for data in train_ds:
        validate_patient_graph(data)
        nodes.append(data.num_nodes)
        edges.append(data.num_edges)
        if data.num_edges > 0:
            for et in data.edge_type.numpy():
                edge_counts[et] += 1

    print(f"\nMean nodes: {np.mean(nodes):.1f}, Median nodes: {np.median(nodes)}")
    print(f"Mean edges: {np.mean(edges):.1f}, Median edges: {np.median(edges)}")
    print(f"Edge frequencies: {edge_counts}")

    # 6. Specific Tests
    print("\nSpecific Architectural Requirements:")
    
    # Find patient with missing PCT
    for i, data in enumerate(train_ds):
        pct_id = hash("PCT") % 1000
        n_pct = (data.concept_id == pct_id).sum().item()
        if n_pct == 0:
            print(f"  Patient {data.patient_id} has NO PCT measurements.")
            print(f"  PCT Graph Nodes: {n_pct}")
            print(f"  PCT Biomarker Slots (K): {data.missingness_input.size(0)}")
            break

    # Find patient with repeated Lactate
    lac_id = hash("Lactate") % 1000
    for data in train_ds:
        if (data.concept_id == lac_id).sum().item() > 1:
            rec_mask = (data.edge_type == 1)
            if rec_mask.sum() > 0:
                edges_t = data.edge_index[:, rec_mask]
                dt = data.edge_delta_t[rec_mask]
                
                # Filter for Lactate
                is_lac = (data.concept_id[edges_t[0]] == lac_id) & (data.concept_id[edges_t[1]] == lac_id)
                if is_lac.sum() > 0:
                    print(f"\n  Patient {data.patient_id} has repeated Lactate.")
                    print(f"  Recurrence edge delta_t: {dt[is_lac].numpy()}")
                    # Save viz of this one
                    os.makedirs("outputs/figures", exist_ok=True)
                    visualize_patient_graph(data, "outputs/figures/example_patient_graph.png")
                    print("  Saved: outputs/figures/example_patient_graph.png")
                    break

    print("\nPhase 4 verification completed successfully.")

if __name__ == "__main__":
    main()
