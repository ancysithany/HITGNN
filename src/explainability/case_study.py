"""
src.explainability.case_study - Orchestrate explainability for a single patient.
"""
from typing import Dict, Any, List
import json
import os

from .attention_explainer import extract_hierarchical_attention
from .gradient_explainer import compute_gradient_x_input
from .shap_explainer import extract_shap_metadata, extract_shap_missingness

def generate_patient_case_study(
    patient,
    batch,
    model,
    train_metadata_bg,
    train_miss_bg,
    biomarker_names,
    reference_embeddings=None,
    reference_metadata=None,
    reference_ids=None
) -> Dict[str, Any]:
    """
    Runs all explainability modules for a given patient.
    """
    # 1. Attention
    attention_results = extract_hierarchical_attention(
        patient, batch, model,
        reference_embeddings=reference_embeddings,
        reference_metadata=reference_metadata,
        reference_ids=reference_ids
    )
    
    # 2. Gradient x Input
    gradient_results = compute_gradient_x_input(
        batch, model,
        reference_embeddings=reference_embeddings,
        reference_metadata=reference_metadata,
        reference_ids=reference_ids
    )
    
    # 3. SHAP Metadata
    shap_meta = extract_shap_metadata(
        batch, model, train_metadata_bg,
        reference_embeddings=reference_embeddings,
        reference_metadata=reference_metadata,
        reference_ids=reference_ids
    )
    
    # 4. SHAP Missingness
    shap_miss = extract_shap_missingness(
        batch, model, train_miss_bg, biomarker_names,
        reference_embeddings=reference_embeddings,
        reference_metadata=reference_metadata,
        reference_ids=reference_ids
    )
    
    case_study = {
        "patient_id": patient.patient_id,
        "true_label": patient.mortality,
        "prediction_logit": attention_results["logit"],
        "attention": {
            "type_importance": attention_results["type_importance"],
            "event_importance": attention_results["event_importance"],
            "population_neighbors": attention_results["population_neighbors"]
        },
        "gradient": {
            "metadata": gradient_results["metadata_attribution"]
            # raw missingness grad is not serialized to json for size
        },
        "shap": {
            "metadata": shap_meta,
            "missingness": shap_miss
        }
    }
    
    return case_study

def serialize_case_study(case_study: Dict[str, Any], output_dir: str):
    """
    Saves case study to JSON and CSV formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    pid = case_study["patient_id"]
    
    # JSON
    # Remove raw shap tensors from dict before json dump
    if "raw_shap_tensor" in case_study["shap"]["missingness"]:
        del case_study["shap"]["missingness"]["raw_shap_tensor"]
        
    json_path = os.path.join(output_dir, f"{pid}.json")
    with open(json_path, 'w') as f:
        json.dump(case_study, f, indent=2)
        
    # CSVs for SHAP
    import pandas as pd
    
    # Metadata SHAP
    meta_df = pd.DataFrame.from_dict(
        case_study["shap"]["metadata"]["shap_values"], 
        orient='index'
    )
    meta_df.index.name = "feature"
    meta_df.to_csv(os.path.join(output_dir, f"shap_metadata_{pid}.csv"))
    
    # Missingness SHAP
    miss_df = pd.DataFrame.from_dict(
        case_study["shap"]["missingness"]["shap_values"],
        orient='index'
    )
    miss_df.index.name = "feature"
    miss_df.to_csv(os.path.join(output_dir, f"shap_missingness_{pid}.csv"))
