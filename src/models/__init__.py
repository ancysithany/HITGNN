"""
src.models — HITGNN neural network components.

Modules:
    - time_encoding: Relative time encoding φ(Δt)
    - missingness_encoder: Missingness-aware biomarker encoder
    - te_gat: Temporal Edge-Aware Graph Attention Network
    - hierarchical_pool: Two-stage type → patient attention pooling
    - prediction_head: Final logit prediction (sigmoid at inference only)
    - hitgnn: Full model assembly
"""


def __getattr__(name: str):
    """Lazy import to avoid cascading dependency errors."""
    _imports = {
        "TimeEncoding": ("src.models.time_encoding", "TimeEncoding"),
        "MissingnessEncoder": ("src.models.missingness_encoder", "MissingnessEncoder"),
        "TEGATLayer": ("src.models.te_gat", "TEGATLayer"),
        "TEGATStack": ("src.models.te_gat", "TEGATStack"),
        "HierarchicalPooling": ("src.models.hierarchical_pool", "HierarchicalPooling"),
        "PredictionHead": ("src.models.prediction_head", "PredictionHead"),
        "HITGNN": ("src.models.hitgnn", "HITGNN"),
    }
    if name in _imports:
        module_path, attr = _imports[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")


__all__ = [
    "TimeEncoding",
    "MissingnessEncoder",
    "TEGATLayer",
    "TEGATStack",
    "HierarchicalPooling",
    "PredictionHead",
    "HITGNN",
]
