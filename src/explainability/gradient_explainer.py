"""
src.explainability.gradient_explainer - Gradient x Input attribution.
"""
import torch

def compute_gradient_x_input(
    batch,
    model,
    reference_embeddings=None,
    reference_metadata=None,
    reference_ids=None
):
    """
    Computes Gradient x Input attribution for continuous features.
    Target is the raw mortality logit.
    Returns dictionaries of feature importance.
    """
    model.eval()
    
    # We must clone inputs that we want gradients for
    batch_clone = batch.clone()
    
    meta_in = batch_clone.patient_meta.clone().detach().requires_grad_(True)
    miss_in = batch_clone.missingness_input.clone().detach().requires_grad_(True)
    
    batch_clone.patient_meta = meta_in
    batch_clone.missingness_input = miss_in
    
    # Enable grad context
    with torch.enable_grad():
        if reference_embeddings is not None:
            # We don't need gradients for reference variables right now
            logit = model(
                batch_clone,
                reference_embeddings=reference_embeddings,
                reference_metadata=reference_metadata,
                reference_ids=reference_ids
            )
        else:
            logit = model(batch_clone)
            
        # Target is the logit itself (assuming batch size 1)
        logit.backward()
        
    # Gradient x Input
    meta_attr = (meta_in.grad * meta_in).detach().squeeze(0) # [10]
    miss_attr = (miss_in.grad * miss_in).detach().squeeze(0) # [6, 7, 4]
    
    meta_names = [
        "age_norm", "sofa_norm", "icu_MICU", "icu_SICU", "icu_CCU",
        "diabetes", "CKD", "liver_disease", "heart_failure", "COPD"
    ]
    
    meta_importance = {}
    for i, name in enumerate(meta_names):
        meta_importance[name] = meta_attr[i].item()
        
    return {
        "logit": logit.item(),
        "metadata_attribution": meta_importance,
        "missingness_attribution": miss_attr,
        "missingness_grad": miss_in.grad.detach().squeeze(0)
    }
