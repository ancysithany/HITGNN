"""
scripts/run_missingness_robustness.py

Evaluates the FULL_HITGNN Phase 8 model under different synthetic
missingness mechanisms applied at test time.
"""

import os
import sys
from pathlib import Path
import json
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.hitgnn import PopulationHITGNN
from src.training.population_trainer import PopulationTrainer
from src.data.factory import create_dataset_adapter
from src.preprocessing.pipeline import build_hitgnn_datasets

def main():
    config = load_config("configs/default.yaml")
    seed = config.get("seed", 42)
    set_seed(seed)
    
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    print(f"Using device: {device}")
    
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    
    # 1. Load FULL_HITGNN
    full_config = load_config("configs/default.yaml")
    full_config["ablation"] = {}
    
    ckpt_path = "outputs/checkpoints/best_population_model.pt"
    if not os.path.exists(ckpt_path):
        print("ERROR: FULL_HITGNN checkpoint missing!")
        return
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    train_ds, val_ds, test_ds, vocab = build_hitgnn_datasets(cohort, full_config)
    test_ds.meta_norm.load_state_dict(checkpoint.get('meta_normalizer_state', {}))
    
    full_model = PopulationHITGNN(full_config, vocab.size)
    full_model.load_state_dict(checkpoint['model_state_dict'])
    
    train_loader = DataLoader(train_ds, batch_size=full_config.get("training", {}).get("batch_size", 32), shuffle=False)
    trainer = PopulationTrainer(full_model, train_loader, None, full_config, device, test_ds.meta_norm)
    ref_emb, ref_meta, ref_ids = trainer.build_reference_bank()
    
    scenarios = ["MCAR", "MAR", "INFORMATIVE"]
    rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    results = []
    
    for scenario in scenarios:
        for rate in rates:
            print(f"Evaluating Scenario: {scenario}, Rate: {rate}")
            
            # Apply masking to test set
            test_ds.masking_config = {
                "scenario": scenario,
                "rate": rate,
                "seed": seed
            }
            
            test_loader = DataLoader(test_ds, batch_size=full_config.get("training", {}).get("batch_size", 32), shuffle=False)
            _, metrics, *_ = trainer.evaluate(test_loader, ref_emb, ref_meta, ref_ids)
            
            results.append({
                "scenario": scenario,
                "rate": rate,
                "auroc": metrics['auroc'],
                "auprc": metrics['auprc'],
                "f1": metrics['f1'],
                "brier_score": metrics['brier_score']
            })
            
    df = pd.DataFrame(results)
    df.to_csv("outputs/metrics/missingness_robustness_results.csv", index=False)
    
    # 2. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"MCAR": "blue", "MAR": "orange", "INFORMATIVE": "red"}
    markers = {"MCAR": "o", "MAR": "s", "INFORMATIVE": "^"}
    
    for scenario in scenarios:
        subset = df[df["scenario"] == scenario]
        axes[0].plot(subset["rate"], subset["auroc"], label=scenario, 
                     color=colors[scenario], marker=markers[scenario], linewidth=2, markersize=8)
        axes[1].plot(subset["rate"], subset["auprc"], label=scenario,
                     color=colors[scenario], marker=markers[scenario], linewidth=2, markersize=8)
                     
    # Formatting AUROC
    axes[0].set_title("Robustness to Missingness: AUROC")
    axes[0].set_xlabel("Induced Missingness Rate (Biomarkers Only)")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Formatting AUPRC
    axes[1].set_title("Robustness to Missingness: AUPRC")
    axes[1].set_xlabel("Induced Missingness Rate (Biomarkers Only)")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    fig.suptitle("Results: Missingness Robustness", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    out_path = "outputs/figures/results_missingness_robustness.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
