"""
src.preprocessing — Data normalization and temporal binning utilities.
"""


def __getattr__(name: str):
    """Lazy import to avoid cascading dependency errors."""
    if name == "FeatureNormalizer":
        from src.preprocessing.normalizer import FeatureNormalizer
        return FeatureNormalizer
    elif name == "TemporalBinner":
        from src.preprocessing.temporal_binner import TemporalBinner
        return TemporalBinner
    raise AttributeError(f"module 'src.preprocessing' has no attribute {name!r}")


__all__ = ["FeatureNormalizer", "TemporalBinner"]
