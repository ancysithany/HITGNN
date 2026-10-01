"""
scripts/evaluate_population.py — Phase 8 Population Model Evaluation.
"""

import sys
import os
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import HITGNNPatientDataset

from src.data.vocab import ConceptVocabulary
from src.data.terminology import ClinicalTerminology
from src.models.hitgnn import PopulationHITGNN
from src.training.population_trainer import PopulationTrainer
from src.training.metrics import find_best_threshold

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_population_model.pt")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    
    device_str = config.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu")
    
    from src.data.factory import create_dataset_adapter
    from src.preprocessing.pipeline import build_hitgnn_datasets
    
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    
    # Checkpoint restore
    ckpt = torch.load(args.checkpoint, map_location=device)
    meta_norm_state = ckpt.get('meta_normalizer_state', None)
    
    train_dataset, val_dataset, test_dataset, vocab = build_hitgnn_datasets(cohort, config, meta_norm_state=meta_norm_state)
    
    # We must use shuffle=False for predictable neighbor indices mapping
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = PopulationHITGNN(config, vocab_size=vocab.size)
    model.load_state_dict(ckpt['model_state_dict'])
    
    trainer = PopulationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        meta_normalizer=train_dataset.meta_norm
    )
    
    # 1. Rebuild Reference Bank (eval mode, no grad)
    ref_emb, ref_meta, ref_ids = trainer.build_reference_bank()
    
    # 2. Evaluate Validation (find threshold)
    val_loss, val_metrics, val_y, val_prob, val_logits, _, _, _, _ = trainer.evaluate(
        val_loader, ref_emb, ref_meta, ref_ids
    )
    
    threshold = find_best_threshold(val_y, val_prob, strategy="validation_f1")
    print(f"Selected Validation Threshold: {threshold:.4f}")
    
    # 3. Evaluate Test
    (test_loss, test_metrics, test_y, test_prob, test_logits,
     test_qids, test_neigh_idx, test_neigh_sim, test_neigh_attn) = trainer.evaluate(
        test_loader, ref_emb, ref_meta, ref_ids, threshold=threshold
    )
    
    print("\n--- TEST SET METRICS ---")
    for k, v in test_metrics.items():
        print(f"{k.upper()}: {v:.4f}")
        
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/population_model_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    df_preds = pd.DataFrame({
        "patient_id": test_qids,
        "true_label": test_y,
        "logit": test_logits,
        "probability": test_prob,
        "predicted_label": (test_prob >= threshold).astype(int)
    })
    df_preds.to_csv("outputs/metrics/population_test_predictions.csv", index=False)
    
    # Generate neighbor CSV
    # test_neigh_idx: [Q, k]
    # test_neigh_sim: [Q, k]
    # test_neigh_attn: [Q, k]
    
    rows = []
    for i, qid in enumerate(test_qids):
        for rank in range(test_neigh_idx.size(1)):
            idx = test_neigh_idx[i, rank].item()
            sim = test_neigh_sim[i, rank].item()
            attn = test_neigh_attn[i, rank].item()
            r_id = ref_ids[idx]
            
            rows.append({
                "query_patient_id": qid,
                "query_split": "TEST",
                "neighbor_rank": rank + 1,
                "neighbor_patient_id": r_id,
                "similarity_score": sim,
                "attention_weight": attn
            })
            
    df_neigh = pd.DataFrame(rows)
    df_neigh.to_csv("outputs/metrics/population_neighbors.csv", index=False)
    
if __name__ == "__main__":
    main()
