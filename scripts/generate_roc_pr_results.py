"""
scripts/generate_roc_pr_results.py

Generates ROC and Precision-Recall curves comparing Patient-only HITGNN (Phase 7)
and Population HITGNN (Phase 8).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score

def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    
    # Paths to existing predictions
    pat_csv = "outputs/metrics/test_predictions.csv"
    pop_csv = "outputs/metrics/population_test_predictions.csv"
    
    if not os.path.exists(pat_csv) or not os.path.exists(pop_csv):
        print("Error: Missing prediction CSV files. Run evaluations first.")
        return
        
    df_pat = pd.read_csv(pat_csv)
    df_pop = pd.read_csv(pop_csv)
    
    # Inner join on patient_id to ensure exact match
    df = pd.merge(df_pat, df_pop, on="patient_id", suffixes=("_pat", "_pop"))
    
    # Verify labels match exactly
    assert np.allclose(df["true_label_pat"], df["true_label_pop"]), "Mismatched true labels"
    y_true = df["true_label_pat"].values
    
    prob_pat = df["probability_pat"].values
    prob_pop = df["probability_pop"].values
    
    # Compute metrics
    auroc_pat = roc_auc_score(y_true, prob_pat)
    auprc_pat = average_precision_score(y_true, prob_pat)
    
    auroc_pop = roc_auc_score(y_true, prob_pop)
    auprc_pop = average_precision_score(y_true, prob_pop)
    
    fpr_pat, tpr_pat, _ = roc_curve(y_true, prob_pat)
    prec_pat, rec_pat, _ = precision_recall_curve(y_true, prob_pat)
    
    fpr_pop, tpr_pop, _ = roc_curve(y_true, prob_pop)
    prec_pop, rec_pop, _ = precision_recall_curve(y_true, prob_pop)
    
    # Save metrics
    metrics = []
    metrics.append({
        "model": "Patient-only HITGNN",
        "auroc": auroc_pat,
        "auprc": auprc_pat,
        "n_test": len(df)
    })
    metrics.append({
        "model": "Population HITGNN",
        "auroc": auroc_pop,
        "auprc": auprc_pop,
        "n_test": len(df)
    })
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("outputs/metrics/roc_pr_comparison.csv", index=False)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Results", fontsize=14, y=1.05)
    
    # Panel (a): ROC Curve
    ax = axes[0]
    ax.plot(fpr_pat, tpr_pat, label=f"Patient-only HITGNN (AUROC={auroc_pat:.3f})", color='blue', linestyle='--')
    ax.plot(fpr_pop, tpr_pop, label=f"Population HITGNN (AUROC={auroc_pop:.3f})", color='red', linestyle='-')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(False)
    
    # Panel (b): PR Curve
    ax = axes[1]
    ax.plot(rec_pat, prec_pat, label=f"Patient-only HITGNN (AUPRC={auprc_pat:.3f})", color='blue', linestyle='--')
    ax.plot(rec_pop, prec_pop, label=f"Population HITGNN (AUPRC={auprc_pop:.3f})", color='red', linestyle='-')
    # Baseline for PR is prevalence
    prevalence = np.mean(y_true)
    ax.axhline(prevalence, color='k', linestyle='--', alpha=0.5, label=f"Random (AUPRC={prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig("outputs/figures/results_roc_pr_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved outputs/figures/results_roc_pr_comparison.png")
    print("Saved outputs/metrics/roc_pr_comparison.csv")

if __name__ == "__main__":
    main()
