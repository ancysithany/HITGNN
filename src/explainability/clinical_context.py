"""
src.explainability.clinical_context — Clinical Terminology Contextualization (Level 4).

Maps influential clinical concepts (from Levels 1–3) to semantic groups:
    - Laboratory Finding
    - Clinical Finding
    - Pharmacologic Substance
    - Procedure
    - Diagnostic Concept

Uses the simplified terminology dictionary from ``src.data.terminology``.

This is ASSOCIATIONAL contextualization.
Never describe as biological pathway inference.
"""

from typing import Dict, List

from src.data.terminology import ClinicalTerminology
from src.data.vocab import ConceptVocabulary


class ClinicalContextualizer:
    """Map attributed concepts to clinical semantic groups.

    Parameters
    ----------
    vocab : ConceptVocabulary
        Canonical concept vocabulary.
    terminology : ClinicalTerminology
        Semantic relationship dictionary.
    """

    def __init__(
        self,
        vocab: ConceptVocabulary,
        terminology: ClinicalTerminology,
    ) -> None:
        # TODO: Implement in Phase 7 (explainability).
        raise NotImplementedError("Implement in Phase 7")

    def contextualize(
        self,
        concept_ids: List[int],
        attribution_scores: List[float],
    ) -> Dict[str, List[Dict]]:
        """Group attributed concepts by semantic category.

        Parameters
        ----------
        concept_ids : List[int]
            Concept IDs of the most attributed events.
        attribution_scores : List[float]
            Corresponding attribution scores.

        Returns
        -------
        Dict[str, List[Dict]]
            Mapping from semantic group to list of dicts:
            ``{"concept_name": str, "attribution": float, "relations": [...]}``.
        """
        raise NotImplementedError("Implement in Phase 7")
