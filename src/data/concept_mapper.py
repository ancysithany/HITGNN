"""
src.data.concept_mapper — Canonical concept mappings.
"""
from typing import Optional

class BaseConceptMapper:
    """Base interface for mapping raw concepts to canonical vocabulary."""
    def map_concept(self, source_concept: str) -> Optional[str]:
        raise NotImplementedError

class SyntheticConceptMapper(BaseConceptMapper):
    """Identity mapper for synthetic data."""
    def map_concept(self, source_concept: str) -> Optional[str]:
        return source_concept
