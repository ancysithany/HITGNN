"""
scripts/train.py — Phase 7 Training Script.
"""

import sys
import argparse
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.dataset import HITGNNPatientDataset
from src.models.hitgnn import PatientMortalityModel
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.epochs is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["epochs"] = args.epochs
        
    device_str = args.device if args.device else config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    print(f"Using device: {device}")
    
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from src.data.factory import create_dataset_adapter
    from src.preprocessing.pipeline import build_hitgnn_datasets
    
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    
    train_ds, val_ds, _, vocab = build_hitgnn_datasets(cohort, config)
    
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PatientMortalityModel(config, vocab_size=vocab.size)
    
    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    trainer.train()

if __name__ == "__main__":
    main()
