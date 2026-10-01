"""
src.graph — Graph construction modules.

Provides:
    - Per-patient temporal event graph construction
    - Inter-patient population graph (inductive k-NN)
"""


def __getattr__(name: str):
    """Lazy import to avoid cascading dependency errors."""
    if name == "PatientGraphBuilder":
        from src.graph.graph_builder import PatientGraphBuilder
        return PatientGraphBuilder
    elif name == "PopulationGraphBuilder":
        from src.graph.population_graph import PopulationGraphBuilder
        return PopulationGraphBuilder
    raise AttributeError(f"module 'src.graph' has no attribute {name!r}")


__all__ = ["PatientGraphBuilder", "PopulationGraphBuilder"]
