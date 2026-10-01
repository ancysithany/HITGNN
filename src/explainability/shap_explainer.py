"""
src.explainability.shap_explainer - SHAP integration.
"""
import torch
import numpy as np
import shap
from .shap_wrappers import MetadataSHAPWrapper, MissingnessSHAPWrapper

def extract_shap_metadata(
    batch,
    model,
    train_metadata_bg,
    reference_embeddings=None,
    reference_metadata=None,
    reference_ids=None
):
    """
    Computes SHAP values for patient metadata using KernelExplainer.
    
    train_metadata_bg: numpy array of shape [num_bg, 10]
    """
    wrapper = MetadataSHAPWrapper(
        model, batch, reference_embeddings, reference_metadata, reference_ids
    )
    wrapper.eval()
    
    # Define a predict function that takes numpy array and returns numpy array
    def predict_fn(x):
        # x is [B, 10]
        x_tensor = torch.tensor(x, dtype=torch.float32, device=batch.patient_meta.device)
        with torch.no_grad():
            preds = wrapper(x_tensor)
        return preds.cpu().numpy()
        
    explainer = shap.KernelExplainer(predict_fn, train_metadata_bg)
    
    x_eval = batch.patient_meta.cpu().numpy()
    
    # Explain the instance
    shap_values = explainer.shap_values(x_eval, nsamples=100)
    
    # SHAP sometimes returns a list of arrays (one for each class). We just have 1 output logit.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # Get base value
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]
        
    meta_names = [
        "age_norm", "sofa_norm", "icu_MICU", "icu_SICU", "icu_CCU",
        "diabetes", "CKD", "liver_disease", "heart_failure", "COPD"
    ]
    
    results = {}
    for i, name in enumerate(meta_names):
        val = x_eval[0, i]
        s_val = shap_values[0, i]
        results[name] = {
            "feature_value": float(val),
            "shap_value": float(s_val),
            "abs_shap_value": abs(float(s_val))
        }
        
    return {
        "base_value": float(base_value),
        "shap_values": results
    }


def extract_shap_missingness(
    batch,
    model,
    train_miss_bg,
    biomarker_names,
    reference_embeddings=None,
    reference_metadata=None,
    reference_ids=None
):
    """
    Computes SHAP values for missingness using GradientExplainer.
    
    train_miss_bg: torch tensor of shape [num_bg, 6, 7, 4]
    """
    wrapper = MissingnessSHAPWrapper(
        model, batch, reference_embeddings, reference_metadata, reference_ids
    )
    wrapper.eval()
    
    # We use GradientExplainer which requires a PyTorch model
    # It takes inputs and background
    explainer = shap.GradientExplainer(wrapper, train_miss_bg)
    
    x_eval = batch.missingness_input
    
    # SHAP might complain about PyG models if the internal graph structure requires grad (which it doesn't)
    # If GradientExplainer fails, we could fallback to KernelExplainer, but it's slow for 168 features.
    try:
        shap_values = explainer.shap_values(x_eval)
        
        # Estimate base_value by running the background through the wrapper
        with torch.no_grad():
            bg_preds = wrapper(train_miss_bg)
            base_value = bg_preds.mean().item()
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        if shap_values.ndim == 5 and shap_values.shape[-1] == 1:
            shap_values = shap_values.squeeze(-1)
            
    except Exception as e:
        print(f"GradientExplainer failed: {e}. Falling back to KernelExplainer with nsamples=100")
        
        # Fallback to KernelExplainer
        def predict_fn(x):
            # x is [B, 168]
            x_tensor = torch.tensor(x, dtype=torch.float32, device=x_eval.device).view(-1, 6, 7, 4)
            with torch.no_grad():
                preds = wrapper(x_tensor)
            return preds.cpu().numpy()
            
        flat_bg = train_miss_bg.view(train_miss_bg.size(0), -1).cpu().numpy()
        k_explainer = shap.KernelExplainer(predict_fn, flat_bg)
        flat_eval = x_eval.view(1, -1).cpu().numpy()
        shap_values_flat = k_explainer.shap_values(flat_eval, nsamples=100)
        
        if isinstance(shap_values_flat, list):
            shap_values_flat = shap_values_flat[0]
            
        shap_values = shap_values_flat.reshape(1, 6, len(biomarker_names), 4)
        base_value = k_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
            
    # Format output
    # missingness shape: [1, 6, num_biomarkers, 4]
    channels = ["value", "observed", "delta_t", "never_observed"]
    
    results = {}
    
    for b_idx, b_name in enumerate(biomarker_names):
        for w_idx in range(6):
            window_name = f"w{w_idx}"
            for c_idx, c_name in enumerate(channels):
                feat_name = f"{b_name}_{window_name}_{c_name}"
                val = x_eval[0, w_idx, b_idx, c_idx].item()
                s_val = shap_values[0, w_idx, b_idx, c_idx]
                if isinstance(s_val, torch.Tensor):
                    s_val = s_val.item()
                results[feat_name] = {
                    "feature_value": val,
                    "shap_value": float(s_val),
                    "abs_shap_value": abs(float(s_val)),
                    "biomarker": b_name,
                    "window": window_name,
                    "channel": c_name
                }
                
    return {
        "base_value": float(base_value),
        "shap_values": results,
        "raw_shap_tensor": shap_values[0] # [6, 7, 4]
    }
