"""
scripts/capture_regression_snapshot.py
"""
import sys
import os
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.synthetic_generator import SyntheticCohortGenerator, patient_level_split
from src.data.vocab import ConceptVocabulary
from src.data.terminology import ClinicalTerminology
from src.preprocessing.normalizer import FeatureNormalizer
from src.preprocessing.event_normalizer import EventNormalizer
from src.preprocessing.meta_normalizer import PatientMetaNormalizer
from src.preprocessing.temporal_binner import TemporalBinner
from src.graph.graph_builder import PatientGraphBuilder
from src.data.dataset import HITGNNPatientDataset
import numpy as np
import json

def serialize_tensor(t):
    if t is None:
        return None
    return t.detach().cpu().numpy().tolist()

def main():
    config = load_config("configs/default.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)
    
    vocab = ConceptVocabulary.build_default()
    term = ClinicalTerminology.build_default()
    gen = SyntheticCohortGenerator(config, vocab, seed=seed)
    patients = gen.generate()
    
    train_cfg = config.get("training", {})
    split_ratios = train_cfg.get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    ids = [p.patient_id for p in patients]
    labels = [p.mortality for p in patients]
    splits = patient_level_split(ids, split_ratios, seed=seed, labels=labels)
    
    train_patients = [p for p in patients if p.patient_id in splits["train"]]
    biomarkers = config.get("biomarkers", [])
    
    bio_norm = FeatureNormalizer()
    event_norm = EventNormalizer()
    for bm in biomarkers:
        vals = [e.value for p in train_patients for e in p.events if e.event_type == "LAB" and e.concept_name == bm and e.value is not None]
        bio_norm.fit(bm, np.array(vals) if vals else np.array([]))
        
    continuous = biomarkers + ["MAP", "Heart Rate", "Respiratory Rate", "Temperature", "SpO2", "GCS"]
    for c in continuous:
        vals = [e.value for p in train_patients for e in p.events if e.concept_name == c and e.value is not None]
        event_norm.fit(c, np.array(vals) if vals else np.array([]))
        
    meta_norm = PatientMetaNormalizer()
    meta_norm.fit(train_patients)
    
    graph_builder = PatientGraphBuilder(vocab, term, event_norm, config.get("graph", {}))
    binner = TemporalBinner(n_bins=config.get("temporal", {}).get("n_bins", 6))
    
    train_ds = HITGNNPatientDataset(patients, graph_builder, binner, bio_norm, meta_norm, biomarkers, splits["train"])
    
    # We will pick 3 specific patients from the train set, e.g., index 0, 1, 2
    snapshots = {}
    for i in range(3):
        data = train_ds[i]
        pid = train_ds.patient_records[i].patient_id
        
        # original record
        record = next(p for p in patients if p.patient_id == pid)
        
        snapshots[pid] = {
            "target": record.mortality,
            "patient_metadata": record.metadata,
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.size(1),
            "node_type": serialize_tensor(data.node_type),
            "concept_id": serialize_tensor(data.concept_id),
            "edge_index": serialize_tensor(data.edge_index),
            "edge_type": serialize_tensor(data.edge_type),
            "biomarker_masks": serialize_tensor(data.biomarker_masks),
            "missingness_input": serialize_tensor(data.missingness_input),
            "patient_meta": serialize_tensor(data.patient_meta)
        }
    
    os.makedirs("outputs/testing", exist_ok=True)
    with open("outputs/testing/regression_snapshot.json", "w") as f:
        json.dump(snapshots, f, indent=2)
    print(f"Captured snapshot for {list(snapshots.keys())}")

if __name__ == "__main__":
    main()
