"""
scripts/tiny_overfit.py — Tiny Overfit Test for Phase 7.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.vocab import ConceptVocabulary
from src.data.terminology import ClinicalTerminology
from src.data.synthetic_generator import SyntheticCohortGenerator
from src.preprocessing.temporal_binner import TemporalBinner
from src.preprocessing.normalizer import FeatureNormalizer
from src.preprocessing.event_normalizer import EventNormalizer
from src.graph.graph_builder import PatientGraphBuilder
from src.data.dataset import HITGNNPatientDataset
from src.models.hitgnn import PatientMortalityModel

def main():
    config = load_config("configs/default.yaml")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    vocab = ConceptVocabulary.build_default()
    term = ClinicalTerminology.build_default()
    gen = SyntheticCohortGenerator(config, vocab, seed=seed)
    patients = gen.generate()
    
    # Take just 16 patients for tiny overfit
    tiny_patients = patients[:16]
    
    # Need normalizers
    biomarkers = config.get("biomarkers", [])
    bio_norm = FeatureNormalizer()
    event_norm = EventNormalizer()
    
    for bm in biomarkers:
        vals = [e.value for p in tiny_patients for e in p.events
                if e.event_type == "LAB" and e.concept_name == bm and e.value is not None]
        bio_norm.fit(bm, np.array(vals) if vals else np.array([]))
        
    continuous = biomarkers + ["MAP", "Heart Rate", "Respiratory Rate", "Temperature", "SpO2", "GCS"]
    for c in continuous:
        vals = [e.value for p in tiny_patients for e in p.events
                if e.concept_name == c and e.value is not None]
        event_norm.fit(c, np.array(vals) if vals else np.array([]))

    binner = TemporalBinner(n_bins=config.get("temporal", {}).get("n_bins", 6))
    builder = PatientGraphBuilder(vocab, term, event_norm, config.get("graph", {}))
    
    split_ids = set([p.patient_id for p in tiny_patients])
    ds = HITGNNPatientDataset(tiny_patients, builder, binner, bio_norm, biomarkers, split_ids)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    
    model = PatientMortalityModel(config, vocab_size=vocab.size)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    batch = next(iter(loader))
    
    initial_loss = None
    final_loss = None
    
    print("Starting Tiny Overfit...")
    for i in range(100):
        opt.zero_grad()
        logits = model(batch)
        y = batch.y.float()
        
        if logits.ndim == 1:
            logits = logits.view(-1)
            y = y.view(-1)
            
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        
        if i == 0:
            initial_loss = loss.item()
            print(f"Initial Loss: {initial_loss:.4f}")
            
    final_loss = loss.item()
    print(f"Final Loss: {final_loss:.4f}")
    
    if final_loss < initial_loss * 0.5:
        print("TINY OVERFIT: SUCCESS (Loss decreased substantially)")
    else:
        print("TINY OVERFIT: FAILED (Loss did not decrease enough)")

if __name__ == "__main__":
    main()
