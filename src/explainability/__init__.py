"""
src.explainability module
"""
from .attention_explainer import extract_hierarchical_attention
from .gradient_explainer import compute_gradient_x_input
from .shap_wrappers import MetadataSHAPWrapper, MissingnessSHAPWrapper
from .shap_explainer import extract_shap_metadata, extract_shap_missingness
from .case_study import generate_patient_case_study, serialize_case_study

__all__ = [
    "extract_hierarchical_attention",
    "compute_gradient_x_input",
    "MetadataSHAPWrapper",
    "MissingnessSHAPWrapper",
    "extract_shap_metadata",
    "extract_shap_missingness",
    "generate_patient_case_study",
    "serialize_case_study"
]
