"""
src.data.vocab — Canonical Concept ID Vocabulary.

Maps every clinical concept used in HITGNN to a unique integer ID.
Acts as an internal surrogate for LOINC (labs), RxNorm (medications),
ICD/SNOMED (diagnoses), and common vital-sign naming.

This module is intentionally decoupled from real coding systems so that
real MIMIC-IV / eICU preprocessing can later provide a mapping layer
from external codes → these canonical IDs without changing the model.

Concept metadata includes:
    - concept_id (int): unique identifier
    - concept_name (str): human-readable label
    - category (str): one of the six event types
    - unit (str or None): measurement unit for numerical concepts
    - normal_range (tuple or None): (low, high) reference interval
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---- Event type constants ------------------------------------------------

EVENT_TYPES: List[str] = [
    "LAB",
    "VITAL",
    "MEDICATION",
    "PROCEDURE",
    "DIAGNOSIS",
    "CLINICAL_CONCEPT",
]

EVENT_TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(EVENT_TYPES)}


# ---- Concept dataclass ---------------------------------------------------

@dataclass
class ClinicalConcept:
    """Metadata for a single canonical clinical concept."""

    concept_id: int
    concept_name: str
    category: str  # Must be in EVENT_TYPES
    unit: Optional[str] = None
    normal_range: Optional[Tuple[float, float]] = None


# ---- Vocabulary class ----------------------------------------------------

class ConceptVocabulary:
    """Registry of all canonical clinical concepts.

    Provides:
        - Integer ID ↔ concept name lookups
        - Category-based filtering
        - Concept metadata access

    The vocabulary is populated via :meth:`build_default` which creates
    the standard demo concept set, or manually via :meth:`add_concept`.

    Future integration point:
        Real data pipelines should map external codes (LOINC, RxNorm, ICD)
        to these canonical IDs through a separate mapping layer.
    """

    def __init__(self) -> None:
        self._concepts: Dict[int, ClinicalConcept] = {}
        self._name_to_id: Dict[str, int] = {}
        self._next_id: int = 0

    # -- Construction -------------------------------------------------------

    def add_concept(
        self,
        name: str,
        category: str,
        unit: Optional[str] = None,
        normal_range: Optional[Tuple[float, float]] = None,
    ) -> int:
        """Register a new clinical concept and return its integer ID.

        Parameters
        ----------
        name : str
            Human-readable concept name (e.g., ``"Lactate"``).
        category : str
            Event type category.  Must be one of :data:`EVENT_TYPES`.
        unit : str, optional
            Measurement unit (e.g., ``"mmol/L"``).
        normal_range : tuple of float, optional
            ``(low, high)`` reference interval.

        Returns
        -------
        int
            The unique concept ID assigned to this concept.

        Raises
        ------
        ValueError
            If ``category`` is not in :data:`EVENT_TYPES` or the concept
            name is already registered.
        """
        if category not in EVENT_TYPES:
            raise ValueError(
                f"Unknown category '{category}'. Must be one of {EVENT_TYPES}"
            )
        if name in self._name_to_id:
            raise ValueError(f"Concept '{name}' already registered.")

        cid = self._next_id
        self._next_id += 1
        concept = ClinicalConcept(
            concept_id=cid,
            concept_name=name,
            category=category,
            unit=unit,
            normal_range=normal_range,
        )
        self._concepts[cid] = concept
        self._name_to_id[name] = cid
        return cid

    # -- Lookups ------------------------------------------------------------

    def get_by_id(self, concept_id: int) -> ClinicalConcept:
        """Return concept metadata by integer ID."""
        return self._concepts[concept_id]

    def get_by_name(self, name: str) -> ClinicalConcept:
        """Return concept metadata by name."""
        return self._concepts[self._name_to_id[name]]

    def name_to_id(self, name: str) -> int:
        """Return the integer ID for a concept name."""
        return self._name_to_id[name]

    def id_to_name(self, concept_id: int) -> str:
        """Return the human-readable name for a concept ID."""
        return self._concepts[concept_id].concept_name

    def get_concepts_by_category(self, category: str) -> List[ClinicalConcept]:
        """Return all concepts belonging to a given event-type category."""
        return [c for c in self._concepts.values() if c.category == category]

    @property
    def size(self) -> int:
        """Total number of registered concepts."""
        return len(self._concepts)

    # -- Default vocabulary -------------------------------------------------

    @classmethod
    def build_default(cls) -> "ConceptVocabulary":
        """Construct the standard HITGNN demo concept vocabulary.

        Contains 29 clinical concepts across 6 event categories:
            - 7 LAB concepts with units and normal ranges
            - 6 VITAL concepts
            - 3 MEDICATION concepts
            - 5 DIAGNOSIS concepts
            - 3 PROCEDURE concepts
            - 5 CLINICAL_CONCEPT concepts

        Returns
        -------
        ConceptVocabulary
            A fully populated vocabulary.
        """
        vocab = cls()

        # --- Labs (7) ---
        vocab.add_concept("Lactate", "LAB", "mmol/L", (0.5, 2.0))
        vocab.add_concept("Creatinine", "LAB", "mg/dL", (0.6, 1.2))
        vocab.add_concept("Bilirubin", "LAB", "mg/dL", (0.1, 1.2))
        vocab.add_concept("Platelets", "LAB", "K/uL", (150.0, 400.0))
        vocab.add_concept("WBC", "LAB", "K/uL", (4.0, 11.0))
        vocab.add_concept("CRP", "LAB", "mg/L", (0.0, 10.0))
        vocab.add_concept("PCT", "LAB", "ng/mL", (0.0, 0.1))

        # --- Vitals (6) ---
        vocab.add_concept("MAP", "VITAL", "mmHg", (70.0, 100.0))
        vocab.add_concept("HeartRate", "VITAL", "bpm", (60.0, 100.0))
        vocab.add_concept("RespiratoryRate", "VITAL", "/min", (12.0, 20.0))
        vocab.add_concept("Temperature", "VITAL", "C", (36.5, 37.5))
        vocab.add_concept("SpO2", "VITAL", "%", (95.0, 100.0))
        vocab.add_concept("GCS", "VITAL", "points", (15.0, 15.0))

        # --- Medications (3) ---
        vocab.add_concept("Norepinephrine", "MEDICATION", "mcg/kg/min", None)
        vocab.add_concept("Vasopressin", "MEDICATION", "units/min", None)
        vocab.add_concept("Antibiotics", "MEDICATION", None, None)

        # --- Diagnoses (5) ---
        vocab.add_concept("Sepsis", "DIAGNOSIS", None, None)
        vocab.add_concept("Pneumonia", "DIAGNOSIS", None, None)
        vocab.add_concept("UTI", "DIAGNOSIS", None, None)
        vocab.add_concept("AKI", "DIAGNOSIS", None, None)
        vocab.add_concept("ARDS", "DIAGNOSIS", None, None)

        # --- Procedures (3) ---
        vocab.add_concept("Intubation", "PROCEDURE", None, None)
        vocab.add_concept("CentralLine", "PROCEDURE", None, None)
        vocab.add_concept("Dialysis", "PROCEDURE", None, None)

        # --- Clinical Concepts (5) ---
        vocab.add_concept("SepticShock", "CLINICAL_CONCEPT", None, None)
        vocab.add_concept("OrganDysfunction", "CLINICAL_CONCEPT", None, None)
        vocab.add_concept("HemodynamicInstability", "CLINICAL_CONCEPT", None, None)
        vocab.add_concept("RenalDysfunction", "CLINICAL_CONCEPT", None, None)
        vocab.add_concept("RespiratoryFailure", "CLINICAL_CONCEPT", None, None)

        return vocab

