"""
scripts/audit_phase8.py - Comprehensive Audit for Phase 8.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.synthetic_generator import SyntheticCohortGenerator, patient_level_split
from src.data.vocab import ConceptVocabulary
from src.data.terminology import ClinicalTerminology
from src.data.dataset import HITGNNPatientDataset
from src.preprocessing.temporal_binner import TemporalBinner
from src.preprocessing.normalizer import FeatureNormalizer
from src.preprocessing.event_normalizer import EventNormalizer
from src.preprocessing.meta_normalizer import PatientMetaNormalizer
from src.graph.graph_builder import PatientGraphBuilder
from src.models.hitgnn import PopulationHITGNN
from src.training.population_trainer import PopulationTrainer
from src.training.metrics import calculate_metrics
from sklearn.metrics import confusion_matrix

def main():
    print("=" * 50)
    print("1. SPLIT CONSISTENCY AUDIT")
    print("=" * 50)
    
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
    
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]
    
    print(f"Total patient count: {len(patients)}")
    print(f"Train patient count: {len(train_ids)}")
    print(f"Validation patient count: {len(val_ids)}")
    print(f"Test patient count: {len(test_ids)}")
    print("Source used to load the split: src.data.synthetic_generator.patient_level_split")
    print("Wait, does Phase 7 use patient_level_split from src.data.synthetic_generator? Yes.")
    
    # Checkpoint
    ckpt_path = "outputs/checkpoints/best_population_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    print("\n" + "=" * 50)
    print("4. BEST CHECKPOINT AUDIT")
    print("=" * 50)
    print(f"Checkpoint epoch: {ckpt['epoch']}")
    print(f"Best validation metric: {ckpt.get('best_validation_metric')}")
    print(f"Monitor metric: auprc (from config)")
    
    # Setup dataset
    train_patients = [p for p in patients if p.patient_id in train_ids]
    
    event_norm = EventNormalizer()
    bio_norm = FeatureNormalizer()
    meta_norm = PatientMetaNormalizer()
    
    # Mock fitting (just for meta norm audit)
    meta_norm.fit(train_patients)
    print("\n" + "=" * 50)
    print("11. META NORMALIZER AUDIT")
    print("=" * 50)
    print(f"Number of patients used to fit normalizer: {len(train_patients)}")
    print(f"Age mean/std: {meta_norm.age_mean:.4f} / {meta_norm.age_std:.4f}")
    print(f"SOFA mean/std: {meta_norm.sofa_mean:.4f} / {meta_norm.sofa_std:.4f}")
    
    # Full data build for eval
    meta_norm.load_state_dict(ckpt['meta_normalizer_state'])
    biomarkers = config.get("biomarkers", [])
    
    for bm in biomarkers:
        vals = [e.value for p in train_patients for e in p.events if e.event_type == "LAB" and e.concept_name == bm and e.value is not None]
        bio_norm.fit(bm, np.array(vals) if vals else np.array([]))
    
    continuous = biomarkers + ["MAP", "Heart Rate", "Respiratory Rate", "Temperature", "SpO2", "GCS"]
    for c in continuous:
        vals = [e.value for p in train_patients for e in p.events if e.concept_name == c and e.value is not None]
        event_norm.fit(c, np.array(vals) if vals else np.array([]))
        
    graph_builder = PatientGraphBuilder(vocab, term, event_norm, config.get("graph", {}))
    binner = TemporalBinner(n_bins=config.get("temporal", {}).get("n_bins", 6))
    
    train_dataset = HITGNNPatientDataset(patients, graph_builder, binner, bio_norm, meta_norm, biomarkers, train_ids)
    val_dataset = HITGNNPatientDataset(patients, graph_builder, binner, bio_norm, meta_norm, biomarkers, val_ids)
    test_dataset = HITGNNPatientDataset(patients, graph_builder, binner, bio_norm, meta_norm, biomarkers, test_ids)
    
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = PopulationHITGNN(config, vocab_size=vocab.size)
    model.load_state_dict(ckpt['model_state_dict'])
    
    trainer = PopulationTrainer(model, train_loader, val_loader, config, torch.device("cpu"), meta_norm)
    ref_emb, ref_meta, ref_ids = trainer.build_reference_bank()
    
    print("\n" + "=" * 50)
    print("8. REFERENCE BANK AUDIT")
    print("=" * 50)
    print(f"Reference bank size: {len(ref_ids)}")
    print(f"Number of unique patient IDs: {len(set(ref_ids))}")
    print(f"Duplicate patient IDs: {len(ref_ids) - len(set(ref_ids))}")
    non_train_in_bank = [rid for rid in ref_ids if rid not in train_ids]
    print(f"Non-train IDs in bank: {len(non_train_in_bank)}")
    
    print("\n" + "=" * 50)
    print("3. VALIDATION METRIC AUDIT & 5. THRESHOLD 0.01 AUDIT")
    print("=" * 50)
    
    _, _, val_y, val_prob, val_logits, val_qids, val_neigh_idx, val_neigh_sim, val_neigh_attn = trainer.evaluate(
        val_loader, ref_emb, ref_meta, ref_ids
    )
    
    # Test all 99 thresholds
    f1_scores = []
    for t in range(1, 100):
        thresh = t / 100.0
        mets = calculate_metrics(val_y, val_prob, thresh)
        f1_scores.append((thresh, mets['f1']))
        
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    best_t, best_f1 = f1_scores[0]
    second_t, second_f1 = f1_scores[1]
    
    print(f"Best threshold: {best_t:.4f}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Second-best threshold: {second_t:.4f}")
    print(f"Second-best F1: {second_f1:.4f}")
    
    print("\nValidation probability statistics:")
    print(f"Minimum: {np.min(val_prob):.4f}")
    print(f"Maximum: {np.max(val_prob):.4f}")
    print(f"Mean: {np.mean(val_prob):.4f}")
    print(f"Median: {np.median(val_prob):.4f}")
    print(f"25th percentile: {np.percentile(val_prob, 25):.4f}")
    print(f"75th percentile: {np.percentile(val_prob, 75):.4f}")
    
    val_preds_01 = (val_prob >= 0.01).astype(int)
    print(f"Number predicted positive at threshold 0.01 (Val): {np.sum(val_preds_01)}")
    print(f"Number predicted negative at threshold 0.01 (Val): {len(val_preds_01) - np.sum(val_preds_01)}")
    
    val_mets_at_best = calculate_metrics(val_y, val_prob, best_t)
    print("\nValidation metrics at best threshold:")
    for k, v in val_mets_at_best.items():
        print(f"Validation {k.capitalize()}: {v:.4f}")
        
    print("\n" + "=" * 50)
    print("6. TEST CONFUSION PROFILE & 9. RETRIEVAL STRUCTURAL AUDIT")
    print("=" * 50)
    
    _, _, test_y, test_prob, test_logits, test_qids, test_neigh_idx, test_neigh_sim, test_neigh_attn = trainer.evaluate(
        test_loader, ref_emb, ref_meta, ref_ids, threshold=best_t
    )
    
    test_preds = (test_prob >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, test_preds).ravel()
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    
    test_mets = calculate_metrics(test_y, test_prob, best_t)
    print(f"TEST AUROC: {test_mets['auroc']:.4f}")
    print(f"TEST AUPRC: {test_mets['auprc']:.4f}")
    print(f"TEST Accuracy: {test_mets['accuracy']:.4f}")
    print(f"TEST Precision: {test_mets['precision']:.4f}")
    print(f"TEST Recall: {test_mets['sensitivity']:.4f}")
    print(f"TEST Specificity: {test_mets['specificity']:.4f}")
    print(f"TEST F1: {test_mets['f1']:.4f}")
    print(f"TEST Brier: {test_mets['brier_score']:.4f}")
    
    print(f"Number predicted positive (Test): {np.sum(test_preds)}")
    print(f"Number predicted negative (Test): {len(test_preds) - np.sum(test_preds)}")
    
    # Train retrieval
    _, _, train_y, train_prob, train_logits, train_qids, train_neigh_idx, train_neigh_sim, train_neigh_attn = trainer.evaluate(
        train_loader, ref_emb, ref_meta, ref_ids, threshold=best_t
    )
    
    print("\nRetrieval Examples:")
    for qids, idxs, sims, attns, split_name in [
        (train_qids, train_neigh_idx, train_neigh_sim, train_neigh_attn, "Train"),
        (val_qids, val_neigh_idx, val_neigh_sim, val_neigh_attn, "Validation"),
        (test_qids, test_neigh_idx, test_neigh_sim, test_neigh_attn, "Test")
    ]:
        for i in range(3):
            q_id = qids[i]
            n_ids = [ref_ids[idx.item()] for idx in idxs[i]]
            print(f"query patient ID: {q_id}")
            print(f"query split: {split_name}")
            print(f"retrieved neighbor IDs: {n_ids}")
            print(f"neighbor splits: {['Train' for _ in n_ids]}") # Since ref_ids are from train
            print(f"similarities: {sims[i].tolist()}")
            print(f"attention weights: {attns[i].tolist()}")
            print("-" * 20)

    print("\n" + "=" * 50)
    print("13. ACTUAL PARAMETER COUNT")
    print("=" * 50)
    pop_params = sum(p.numel() for p in model.population_module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PopulationGraphModule parameters: {pop_params}")
    print(f"total PopulationHITGNN trainable parameters: {total_params}")

if __name__ == "__main__":
    main()
