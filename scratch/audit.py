import json
import os
import torch
import numpy as np

print("--- 1. CASE SELECTION LOGIC & 5. SHAP ADDITIVITY ---")
cases = [
    "highest_score_positive",
    "lowest_score_positive",
    "highest_score_negative",
    "lowest_score_negative"
]
threshold = -0.1979 # Phase 8 frozen threshold (from test_metrics.json usually)
try:
    with open("outputs/metrics/test_metrics.json", "r") as f:
        metrics = json.load(f)
        if "best_threshold" in metrics:
            threshold = metrics["best_threshold"]
except:
    pass
print(f"Frozen Threshold: {threshold}")

for c in cases:
    # Find file
    files = [f for f in os.listdir("outputs/explanations") if f.startswith(f"case_study_{c}")]
    if not files:
        print(f"Missing {c}")
        continue
    with open(os.path.join("outputs/explanations", files[0]), "r") as f:
        data = json.load(f)
    print(f"\nCase: {c}")
    logit = data["prediction_logit"]
    prob = data["prediction_probability"]
    true_lbl = data["true_label"]
    pred_lbl = 1 if logit >= threshold else 0
    print(f"  True Label: {true_lbl}, Predicted Label: {pred_lbl}, Prob: {prob:.4f}, Raw Logit: {logit:.4f}")
    
    meta_base = data["shap"]["metadata"]["base_value"]
    meta_sum = sum(v["shap_value"] for v in data["shap"]["metadata"]["shap_values"].values())
    print(f"  Metadata SHAP: base {meta_base:.4f} + sum {meta_sum:.4f} = {meta_base+meta_sum:.4f} | Error: {abs((meta_base+meta_sum)-logit):.4f}")
    
    miss_base = data["shap"]["missingness"]["base_value"]
    miss_sum = sum(v["shap_value"] for v in data["shap"]["missingness"]["shap_values"].values())
    print(f"  Missingness SHAP: base {miss_base:.4f} + sum {miss_sum:.4f} = {miss_base+miss_sum:.4f} | Error: {abs((miss_base+miss_sum)-logit):.4f}")

print("\n--- 14. CHECKPOINT INTEGRITY ---")
pt_patient = os.stat("outputs/checkpoints/best_patient_model.pt").st_mtime
pt_pop = os.stat("outputs/checkpoints/best_population_model.pt").st_mtime
print(f"best_patient_model.pt modified: {pt_patient}")
print(f"best_population_model.pt modified: {pt_pop}")
