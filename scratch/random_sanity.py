import torch
import json
from torch_geometric.loader import DataLoader
import numpy as np
import copy
import sys
sys.path.insert(0, ".")
from src.utils.config import load_config
from src.data.factory import create_dataset_adapter
from src.preprocessing.pipeline import build_hitgnn_datasets
from src.models.hitgnn import PopulationHITGNN
from src.explainability.attention_explainer import extract_hierarchical_attention

print("--- 9. RANDOMIZATION SANITY CHECK ---")
config = load_config("configs/default.yaml")
adapter = create_dataset_adapter(config)
cohort = adapter.prepare()
train_dataset, val_dataset, test_dataset, vocab = build_hitgnn_datasets(cohort, config)

device = torch.device("cpu")
model = PopulationHITGNN(config, vocab.size).to(device)
ckpt = torch.load("outputs/checkpoints/best_population_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Fake reference bank for script speed
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
train_batch = next(iter(train_loader)).to(device)
with torch.no_grad():
    out = model.encoder(train_batch, return_attention=False)
    ref_emb = out.fused_embedding
    ref_meta = train_batch.patient_meta
    ref_ids = train_batch.patient_id

data = test_dataset[0]
batch = next(iter(DataLoader([data], batch_size=1))).to(device)

res_orig = extract_hierarchical_attention(test_dataset.patient_records[0], batch, model, ref_emb, ref_meta, ref_ids)
orig_val = list(res_orig["type_importance"].values())[0]

# Randomize copy
model_rand = copy.deepcopy(model)
for p in model_rand.parameters():
    p.data.normal_(0, 1)
model_rand.eval()

res_rand = extract_hierarchical_attention(test_dataset.patient_records[0], batch, model_rand, ref_emb, ref_meta, ref_ids)
rand_val = list(res_rand["type_importance"].values())[0]

print(f"Original Model First Type Importance: {orig_val:.6f}")
print(f"Randomized Model First Type Importance: {rand_val:.6f}")
if orig_val != rand_val:
    print("SUCCESS: Explanations changed after randomizing parameters.")
else:
    print("FAILURE: Explanations did not change.")
