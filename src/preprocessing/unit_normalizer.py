"""
src.preprocessing.unit_normalizer — Normalizes units for measurements.
"""

class BaseUnitNormalizer:
    """Base interface for unit normalization."""
    def normalize_unit(self, concept_name: str, value: float, source_unit: str) -> float:
        raise NotImplementedError

class SyntheticUnitNormalizer(BaseUnitNormalizer):
    """Identity unit normalizer for synthetic data."""
    def normalize_unit(self, concept_name: str, value: float, source_unit: str) -> float:
        return value
