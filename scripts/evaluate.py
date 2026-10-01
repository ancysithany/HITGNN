"""
scripts/evaluate.py — Phase 7 Evaluation Script.
"""

import sys
import os
import argparse
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.dataset import HITGNNPatientDataset
from src.data.dataset import HITGNNPatientDataset
from src.models.hitgnn import PatientMortalityModel
from src.training.trainer import Trainer
from src.training.metrics import find_best_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='Model')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_prc(y_true, y_prob, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(path)
    plt.close()


def plot_calibration(y_true, y_prob, path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    
    # Load config from checkpoint if possible, otherwise from file
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "config" in ckpt:
        config = ckpt["config"]
    else:
        config = load_config(args.config)
        
    device_str = args.device if args.device else config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # ------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------
    from src.data.factory import create_dataset_adapter
    from src.preprocessing.pipeline import build_hitgnn_datasets
    
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    
    _, val_ds, test_ds, vocab = build_hitgnn_datasets(cohort, config)
    splits = {"test": cohort.test_ids}
    
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Model Setup
    # ------------------------------------------------------------------
    model = PatientMortalityModel(config, vocab_size=vocab.size)
    trainer = Trainer(
        model=model,
        train_loader=val_loader,  # dummy
        val_loader=val_loader,
        config=config,
        device=device
    )
    trainer.load_checkpoint(args.checkpoint)
    
    # ------------------------------------------------------------------
    # Threshold Selection on Validation Set
    # ------------------------------------------------------------------
    _, _, val_y, val_prob, _ = trainer.evaluate(val_loader)
    threshold = find_best_threshold(val_y, val_prob, strategy="validation_f1")
    print(f"Selected Threshold on Validation: {threshold:.4f}")

    # ------------------------------------------------------------------
    # Final Test Evaluation
    # ------------------------------------------------------------------
    _, test_metrics, test_y, test_prob, test_logits = trainer.evaluate(test_loader, threshold=threshold)
    
    print("\n--- TEST SET METRICS ---")
    for k, v in test_metrics.items():
        print(f"{k.upper()}: {v:.4f}")
        
    # ------------------------------------------------------------------
    # Save Artifacts
    # ------------------------------------------------------------------
    eval_cfg = config.get("evaluation", {})
    metrics_dir = eval_cfg.get("metrics_dir", "outputs/metrics")
    figs_dir = eval_cfg.get("output_dir", "outputs/figures")
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    
    # Metrics JSON
    test_metrics['threshold'] = threshold
    with open(os.path.join(metrics_dir, "patient_model_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    # Predictions CSV
    test_pred_labels = (test_prob >= threshold).astype(int)
    test_pids = [p for p in splits["test"]]
    
    df = pd.DataFrame({
        "patient_id": test_pids,
        "true_label": test_y,
        "logit": test_logits,
        "probability": test_prob,
        "predicted_label": test_pred_labels
    })
    df.to_csv(os.path.join(metrics_dir, "test_predictions.csv"), index=False)
    
    # Plots
    plot_roc(test_y, test_prob, os.path.join(figs_dir, "roc_curve.png"))
    plot_prc(test_y, test_prob, os.path.join(figs_dir, "pr_curve.png"))
    plot_cm(test_y, test_pred_labels, os.path.join(figs_dir, "confusion_matrix.png"))
    plot_calibration(test_y, test_prob, os.path.join(figs_dir, "calibration_curve.png"))
    
    print(f"\nArtifacts saved to {metrics_dir} and {figs_dir}")


if __name__ == "__main__":
    main()
