"""
scripts/train_population.py — Phase 8 Population Model Training.
"""

import sys
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.hitgnn import PopulationHITGNN
from src.models.hitgnn import PopulationHITGNN
from src.training.population_trainer import PopulationTrainer

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
    
    from src.data.factory import create_dataset_adapter
    from src.preprocessing.pipeline import build_hitgnn_datasets
    
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    
    train_dataset, val_dataset, _, vocab = build_hitgnn_datasets(cohort, config)
    
    batch_size = config.get("training", {}).get("batch_size", 32)
    
    # We must use shuffle=False or shuffle=True? For reference bank, we collect order.
    # The reference bank builder `build_reference_bank` iterates over train_loader and 
    # concatenates everything. Since it evaluates batches, the order doesn't strictly matter
    # as long as we collect IDs, but let's use shuffle=False for stability. Wait, training needs shuffle.
    # Actually, `build_reference_bank` can use the training loader (which might shuffle) but it stores 
    # `all_ids` correctly aligned with embeddings. So it's fine.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = PopulationHITGNN(config, vocab_size=vocab.size)
    
    trainer = PopulationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        meta_normalizer=train_dataset.meta_norm
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
