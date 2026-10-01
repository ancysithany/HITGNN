"""
src.data — Data generation, vocabulary, and terminology modules.

Provides synthetic ICU patient generation, canonical concept vocabularies,
and a simplified clinical terminology dictionary.

Imports are lazy to avoid cascading dependency errors (e.g., torch_geometric
not installed) when importing individual submodules.
"""


def __getattr__(name: str):
    """Lazy import to avoid cascading torch_geometric dependency."""
    if name == "ConceptVocabulary":
        from src.data.vocab import ConceptVocabulary
        return ConceptVocabulary
    elif name == "ClinicalTerminology":
        from src.data.terminology import ClinicalTerminology
        return ClinicalTerminology
    elif name == "SyntheticCohortGenerator":
        from src.data.synthetic_generator import SyntheticCohortGenerator
        return SyntheticCohortGenerator
    elif name == "HITGNNDataset":
        from src.data.dataset import HITGNNDataset
        return HITGNNDataset
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")


__all__ = [
    "ConceptVocabulary",
    "ClinicalTerminology",
    "SyntheticCohortGenerator",
    "HITGNNDataset",
]
