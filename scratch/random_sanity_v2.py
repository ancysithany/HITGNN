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
from src.explainability.gradient_explainer import compute_gradient_x_input
from src.explainability.shap_explainer import extract_shap_metadata

print("--- 3. RANDOMIZATION SANITY CHECK ---")
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

bg_batch = next(iter(train_loader))
train_meta_bg = bg_batch.patient_meta.numpy()

# 1. Original
res_attn_orig = extract_hierarchical_attention(test_dataset.patient_records[0], batch, model, ref_emb, ref_meta, ref_ids)
res_grad_orig = compute_gradient_x_input(batch, model, ref_emb, ref_meta, ref_ids)
res_shap_orig = extract_shap_metadata(batch, model, train_meta_bg, ref_emb, ref_meta, ref_ids)

attn_vec_orig = np.array(list(res_attn_orig["type_importance"].values()))
grad_vec_orig = np.array(list(res_grad_orig["metadata_attribution"].values()))
shap_vec_orig = np.array([v["shap_value"] for v in res_shap_orig["shap_values"].values()])

# 2. Randomize copy
model_rand = copy.deepcopy(model)
for p in model_rand.parameters():
    p.data.normal_(0, 1)
model_rand.eval()

res_attn_rand = extract_hierarchical_attention(test_dataset.patient_records[0], batch, model_rand, ref_emb, ref_meta, ref_ids)
res_grad_rand = compute_gradient_x_input(batch, model_rand, ref_emb, ref_meta, ref_ids)
res_shap_rand = extract_shap_metadata(batch, model_rand, train_meta_bg, ref_emb, ref_meta, ref_ids)

attn_vec_rand = np.array(list(res_attn_rand["type_importance"].values()))
grad_vec_rand = np.array(list(res_grad_rand["metadata_attribution"].values()))
shap_vec_rand = np.array([v["shap_value"] for v in res_shap_rand["shap_values"].values()])

print(f"ATTENTION L2 Distance: {np.linalg.norm(attn_vec_orig - attn_vec_rand):.6f}")
print(f"GRADIENT ATTRIBUTION L2 Distance: {np.linalg.norm(grad_vec_orig - grad_vec_rand):.6f}")
print(f"SHAP L2 Distance: {np.linalg.norm(shap_vec_orig - shap_vec_rand):.6f}")

