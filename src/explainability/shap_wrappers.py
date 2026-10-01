"""
src.explainability.shap_wrappers - PyTorch wrappers for SHAP analysis.
"""
import torch
import torch.nn as nn

class MetadataSHAPWrapper(nn.Module):
    """
    Wraps model to accept ONLY patient_meta.
    Freezes graph structure, missingness_input, and population reference bank.
    
    If PopulationHITGNN is used, retrieval remains FIXED.
    """
    def __init__(self, model, batch, ref_emb=None, ref_meta=None, ref_ids=None):
        super().__init__()
        self.model = model
        self.batch = batch.clone()
        self.ref_emb = ref_emb
        self.ref_meta = ref_meta
        self.ref_ids = ref_ids
        
    def forward(self, patient_meta: torch.Tensor):
        # patient_meta shape [B, 10]
        # the original batch has size 1, but SHAP may pass B > 1
        # we can't easily batch graphs in this simple wrapper unless we use PyG's Batch.from_data_list.
        # But KernelExplainer handles this by passing numpy arrays.
        # DeepExplainer might pass batched tensors.
        
        # If SHAP passes a batch of metadata, we must replicate the graph.
        B = patient_meta.size(0)
        if B == 1:
            self.batch.patient_meta = patient_meta
            if self.ref_emb is not None:
                out = self.model(self.batch, self.ref_emb, self.ref_meta, self.ref_ids)
            else:
                out = self.model(self.batch)
            return out.view(-1, 1)
            
        else:
            # Replicate sequentially to avoid OOM from huge PyG batch
            preds = []
            for i in range(B):
                # We mutate self.batch temporarily or just clone it
                b_clone = self.batch.clone()
                b_clone.patient_meta = patient_meta[i].unsqueeze(0)
                if self.ref_emb is not None:
                    p = self.model(b_clone, self.ref_emb, self.ref_meta, self.ref_ids)
                else:
                    p = self.model(b_clone)
                preds.append(p)
            return torch.cat(preds, dim=0).view(-1, 1)


class MissingnessSHAPWrapper(nn.Module):
    """
    Wraps model to accept ONLY missingness_input.
    Freezes graph structure, patient_meta, and population reference bank.
    """
    def __init__(self, model, batch, ref_emb=None, ref_meta=None, ref_ids=None):
        super().__init__()
        self.model = model
        self.batch = batch.clone()
        self.ref_emb = ref_emb
        self.ref_meta = ref_meta
        self.ref_ids = ref_ids
        
    def forward(self, missingness_input: torch.Tensor):
        # missingness_input shape [B, 168] (if flattened) or [B, 6, 7, 4]
        # SHAP may pass flattened arrays for KernelExplainer, or unflattened for DeepExplainer.
        if missingness_input.dim() == 2:
            # Reshape back to [B, 6, 7, 4]
            missingness_input = missingness_input.view(-1, 6, 7, 4)
            
        B = missingness_input.size(0)
        if B == 1:
            self.batch.missingness_input = missingness_input
            if self.ref_emb is not None:
                out = self.model(self.batch, self.ref_emb, self.ref_meta, self.ref_ids)
            else:
                out = self.model(self.batch)
            return out.view(-1, 1)
        else:
            preds = []
            for i in range(B):
                b_clone = self.batch.clone()
                b_clone.missingness_input = missingness_input[i].unsqueeze(0)
                if self.ref_emb is not None:
                    p = self.model(b_clone, self.ref_emb, self.ref_meta, self.ref_ids)
                else:
                    p = self.model(b_clone)
                preds.append(p)
            return torch.cat(preds, dim=0).view(-1, 1)

