"""
src.evaluation — Metrics computation and visualization.
"""


def __getattr__(name: str):
    """Lazy import to avoid cascading dependency errors."""
    if name == "compute_all_metrics":
        from src.evaluation.metrics import compute_all_metrics
        return compute_all_metrics
    elif name == "generate_all_plots":
        from src.evaluation.plots import generate_all_plots
        return generate_all_plots
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")


__all__ = ["compute_all_metrics", "generate_all_plots"]
