"""
scripts/run_ablations.py

Trains and evaluates HITGNN ablation variants.
"""

import os
import sys
from pathlib import Path
import json
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.hitgnn import PopulationHITGNN, PatientMortalityModel
from src.training.population_trainer import PopulationTrainer
from src.training.trainer import Trainer # Used for patient-only if needed
from src.data.factory import create_dataset_adapter
from src.preprocessing.pipeline import build_hitgnn_datasets

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    
    # Ablations to retrain
    ablations_to_train = {
        "NO_TEMPORAL_EDGES": {"no_temporal_edges": True},
        "NO_SEMANTIC_EDGES": {"no_semantic_edges": True},
        "NO_MISSINGNESS": {"no_missingness": True},
        "MEAN_POOLING": {"mean_pooling": True},
        "STATIC_GNN": {"static_gnn": True},
    }
    
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    
    results = []
    
    # Ensure directories exist
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    
    # 1. Evaluate FULL_HITGNN (Existing Phase 8 Model)
    print("\n--- Evaluating FULL_HITGNN ---")
    full_config = load_config("configs/default.yaml")
    full_config["ablation"] = {}
    train_ds, val_ds, test_ds, vocab = build_hitgnn_datasets(cohort, full_config)
    full_model = PopulationHITGNN(full_config, vocab.size)
    
    # We must load the meta normalizer state first!
    ckpt_path = "outputs/checkpoints/best_population_model.pt"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        full_model.load_state_dict(checkpoint['model_state_dict'])
        test_ds.meta_norm.load_state_dict(checkpoint.get('meta_normalizer_state', {}))
        
        train_loader = DataLoader(train_ds, batch_size=full_config.get("training", {}).get("batch_size", 32), shuffle=False)
        full_trainer = PopulationTrainer(full_model, train_loader, None, full_config, device, test_ds.meta_norm)
        ref_emb, ref_meta, ref_ids = full_trainer.build_reference_bank()
        
        test_loader = DataLoader(test_ds, batch_size=full_config.get("training", {}).get("batch_size", 32), shuffle=False)
        _, full_metrics, *_ = full_trainer.evaluate(test_loader, ref_emb, ref_meta, ref_ids)
        
        full_auroc = full_metrics['auroc']
        full_auprc = full_metrics['auprc']
        
        results.append({
            "variant": "FULL_HITGNN",
            "best_validation_epoch": checkpoint.get("epoch", -1),
            "best_validation_auroc": -1, # We could extract from history but -1 is fine
            "best_validation_auprc": checkpoint.get("best_validation_metric", -1),
            "test_auroc": full_metrics['auroc'],
            "test_auprc": full_metrics['auprc'],
            "test_f1": full_metrics['f1'],
            "test_brier": full_metrics['brier_score'],
            "parameter_count": count_parameters(full_model),
            "checkpoint_path": ckpt_path
        })
    else:
        print("ERROR: FULL_HITGNN checkpoint missing!")
        full_auroc = 0.0
        full_auprc = 0.0
        
    # 2. Evaluate NO_POPULATION_GRAPH (Existing Phase 7 Model)
    print("\n--- Evaluating NO_POPULATION_GRAPH ---")
    pat_config = load_config("configs/default.yaml")
    pat_config["ablation"] = {}
    pat_train_ds, pat_val_ds, pat_test_ds, pat_vocab = build_hitgnn_datasets(cohort, pat_config)
    pat_model = PatientMortalityModel(pat_config, pat_vocab.size)
    pat_ckpt_path = "outputs/checkpoints/best_patient_model.pt"
    if os.path.exists(pat_ckpt_path):
        pat_checkpoint = torch.load(pat_ckpt_path, map_location=device)
        pat_model.load_state_dict(pat_checkpoint['model_state_dict'])
        pat_meta_state = pat_checkpoint.get('meta_normalizer_state', {})
        if pat_meta_state:
            pat_test_ds.meta_norm.load_state_dict(pat_meta_state)
        
        pat_train_loader = DataLoader(pat_train_ds, batch_size=pat_config.get("training", {}).get("batch_size", 32), shuffle=False)
        pat_trainer = Trainer(pat_model, pat_train_loader, None, pat_config, device)
        pat_test_loader = DataLoader(pat_test_ds, batch_size=pat_config.get("training", {}).get("batch_size", 32), shuffle=False)
        _, pat_metrics, *_ = pat_trainer.evaluate(pat_test_loader)
        
        results.append({
            "variant": "NO_POPULATION_GRAPH",
            "best_validation_epoch": pat_checkpoint.get("epoch", -1),
            "best_validation_auroc": -1, 
            "best_validation_auprc": pat_checkpoint.get("best_validation_metric", -1),
            "test_auroc": pat_metrics['auroc'],
            "test_auprc": pat_metrics['auprc'],
            "test_f1": pat_metrics['f1'],
            "test_brier": pat_metrics['brier_score'],
            "parameter_count": count_parameters(pat_model),
            "checkpoint_path": pat_ckpt_path
        })
    
    # 3. Retrain Structural Ablations
    batch_size = config.get("training", {}).get("batch_size", 32)
    
    for variant, ab_config in ablations_to_train.items():
        print(f"\n--- Training {variant} ---")
        var_config = load_config("configs/default.yaml")
        # For NO_TEMPORAL/SEMANTIC, it's under graph.ablation
        if "no_temporal_edges" in ab_config or "no_semantic_edges" in ab_config:
            var_config.setdefault("graph", {})["ablation"] = ab_config
        else:
            var_config["ablation"] = ab_config
            
        train_ds, val_ds, test_ds, vocab = build_hitgnn_datasets(cohort, var_config)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        model = PopulationHITGNN(var_config, vocab.size)
        
        # We need a new checkpoint path for each variant
        ckpt_name = f"best_{variant.lower()}_model.pt"
        
        trainer = PopulationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=var_config,
            device=device,
            meta_normalizer=train_ds.meta_norm
        )
        # Override save method to use unique filename
        def save_ckpt_override(self=trainer, filename=ckpt_name):
            PopulationTrainer.save_checkpoint(self, filename=filename)
        trainer.save_checkpoint = save_ckpt_override
        
        trainer.train()
        
        # Load best and evaluate
        trainer.load_checkpoint(os.path.join(trainer.checkpoint_dir, ckpt_name))
        ref_emb, ref_meta, ref_ids = trainer.build_reference_bank()
        _, test_metrics, *_ = trainer.evaluate(test_loader, ref_emb, ref_meta, ref_ids)
        
        results.append({
            "variant": variant,
            "best_validation_epoch": trainer.best_epoch,
            "best_validation_auroc": -1, # We monitor auprc
            "best_validation_auprc": trainer.best_metric,
            "test_auroc": test_metrics['auroc'],
            "test_auprc": test_metrics['auprc'],
            "test_f1": test_metrics['f1'],
            "test_brier": test_metrics['brier_score'],
            "parameter_count": count_parameters(model),
            "checkpoint_path": os.path.join(trainer.checkpoint_dir, ckpt_name)
        })
        
    # 4. Save and Plot
    df = pd.DataFrame(results)
    
    # Calculate deltas
    df["delta_auroc_vs_full"] = df["test_auroc"] - full_auroc
    df["delta_auprc_vs_full"] = df["test_auprc"] - full_auprc
    df["delta_f1_vs_full"] = df["test_f1"] - df.loc[df["variant"] == "FULL_HITGNN", "test_f1"].values[0]
    df["delta_brier_vs_full"] = df["test_brier"] - df.loc[df["variant"] == "FULL_HITGNN", "test_brier"].values[0]
    
    # Add SEQUENCE_BASELINE as unavailable
    df = pd.concat([df, pd.DataFrame([{
        "variant": "SEQUENCE_BASELINE",
        "best_validation_epoch": None,
        "best_validation_auroc": None,
        "best_validation_auprc": None,
        "test_auroc": None,
        "test_auprc": None,
        "test_f1": None,
        "test_brier": None,
        "parameter_count": None,
        "checkpoint_path": "Unavailable",
        "delta_auroc_vs_full": None,
        "delta_auprc_vs_full": None,
        "delta_f1_vs_full": None,
        "delta_brier_vs_full": None,
    }])], ignore_index=True)
    
    df.to_csv("outputs/metrics/ablation_results.csv", index=False)
    
    import matplotlib.pyplot as plt
    
    # Order for plotting
    order = [
        "FULL_HITGNN",
        "NO_TEMPORAL_EDGES",
        "NO_MISSINGNESS",
        "NO_POPULATION_GRAPH",
        "NO_SEMANTIC_EDGES",
        "MEAN_POOLING",
        "STATIC_GNN",
        "SEQUENCE_BASELINE"
    ]
    
    # Sort DataFrame according to specific order
    # Drop SEQUENCE_BASELINE for the bar plot values if unavailable, but keep label
    df_plot = df.set_index("variant").reindex(order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(order))
    deltas = df_plot["delta_auroc_vs_full"].values
    
    bars = ax.barh(y_pos, [0 if pd.isna(x) else x for x in deltas], align='center', color='skyblue')
    
    # Add exact values next to bars
    for i, bar in enumerate(bars):
        variant = order[i]
        val = deltas[i]
        if pd.isna(val):
            ax.text(0.01, bar.get_y() + bar.get_height()/2, "Unavailable", va='center', ha='left', color='red')
            continue
            
        sign = "+" if val > 0 else ""
        label = f"{sign}{val:.4f}"
        
        # Position text depending on bar direction
        if val >= 0:
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, label, va='center', ha='left')
        else:
            ax.text(val - 0.001, bar.get_y() + bar.get_height()/2, label, va='center', ha='right')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.invert_yaxis()  # top-to-bottom
    ax.set_xlabel("$\Delta$ AUROC relative to FULL_HITGNN")
    ax.set_title("Results: Ablation Analysis")
    
    # Add vertical line at 0
    ax.axvline(0, color='black', linewidth=0.8)
    
    # Add padding to x-axis
    max_abs = np.nanmax(np.abs(deltas))
    ax.set_xlim(-max_abs - 0.05, max_abs + 0.05)
    
    plt.tight_layout()
    plt.savefig("outputs/figures/results_ablation_bar.png", dpi=300, bbox_inches='tight')
    print("Saved outputs/figures/results_ablation_bar.png")

if __name__ == "__main__":
    main()
