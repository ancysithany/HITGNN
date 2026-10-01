"""
src.data.terminology — Simplified Clinical Terminology Dictionary.

Provides a small, hand-curated semantic relationship dictionary that
approximates UMLS-style relationships between clinical concepts.

This is used for:
    1. Constructing SEMANTIC edges in the patient event graph
    2. Level-4 explainability (clinical terminology contextualization)

The dictionary maps concepts to:
    - Semantic groups (Laboratory Finding, Clinical Finding, etc.)
    - Semantic relationships (e.g., ``is_biomarker_for``, ``treats``)

For the demo, this replaces a full UMLS/SNOMED integration.
Real UMLS CUI mappings can be plugged in later through the same interface.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


# ---- Semantic group constants --------------------------------------------

SEMANTIC_GROUPS: List[str] = [
    "Laboratory Finding",
    "Clinical Finding",
    "Pharmacologic Substance",
    "Procedure",
    "Diagnostic Concept",
    "Organ/Tissue Function",
]


# ---- Relationship types --------------------------------------------------

RELATIONSHIP_TYPES: List[str] = [
    "is_biomarker_for",
    "treats",
    "causes",
    "associated_with",
    "measured_by",
    "indicates",
    "complication_of",
]


# ---- Data structures -----------------------------------------------------

@dataclass
class SemanticRelation:
    """A directed semantic relationship between two clinical concepts.

    Attributes
    ----------
    source_concept : str
        Name of the source concept.
    relation_type : str
        One of :data:`RELATIONSHIP_TYPES`.
    target_concept : str
        Name of the target concept or clinical finding.
    """

    source_concept: str
    relation_type: str
    target_concept: str


class ClinicalTerminology:
    """Simplified clinical terminology and semantic relationship dictionary.

    Provides:
        - Concept → semantic group mapping
        - Pairwise semantic relationships between concepts
        - Lookup methods used by graph builder (semantic edges) and
          explainability (Level 4: clinical contextualization)

    Future integration point:
        Replace :meth:`build_default` with a loader that reads from
        UMLS MRCONSO / MRREL tables or a pre-built concept graph.
    """

    def __init__(self) -> None:
        self._concept_to_group: Dict[str, str] = {}
        self._relations: List[SemanticRelation] = []

    # -- Construction -------------------------------------------------------

    def add_concept_group(self, concept_name: str, semantic_group: str) -> None:
        """Map a concept to its semantic group.

        Parameters
        ----------
        concept_name : str
            Clinical concept name (must match vocabulary).
        semantic_group : str
            One of :data:`SEMANTIC_GROUPS`.
        """
        self._concept_to_group[concept_name] = semantic_group

    def add_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
    ) -> None:
        """Add a directed semantic relationship.

        Parameters
        ----------
        source : str
            Source concept name.
        relation_type : str
            Relationship type from :data:`RELATIONSHIP_TYPES`.
        target : str
            Target concept or clinical finding name.
        """
        self._relations.append(
            SemanticRelation(source, relation_type, target)
        )

    # -- Lookups ------------------------------------------------------------

    def get_semantic_group(self, concept_name: str) -> Optional[str]:
        """Return the semantic group for a concept, or None."""
        return self._concept_to_group.get(concept_name)

    def get_relations_for(self, concept_name: str) -> List[SemanticRelation]:
        """Return all semantic relations where ``concept_name`` is the source."""
        return [r for r in self._relations if r.source_concept == concept_name]

    def get_related_concepts(self, concept_name: str) -> Set[str]:
        """Return the set of concepts related to ``concept_name``."""
        related = set()
        for r in self._relations:
            if r.source_concept == concept_name:
                related.add(r.target_concept)
            elif r.target_concept == concept_name:
                related.add(r.source_concept)
        return related

    def has_relation(self, concept_a: str, concept_b: str) -> bool:
        """Check if any semantic relation exists between two concepts."""
        for r in self._relations:
            if (r.source_concept == concept_a and r.target_concept == concept_b) or \
               (r.source_concept == concept_b and r.target_concept == concept_a):
                return True
        return False

    @property
    def all_relations(self) -> List[SemanticRelation]:
        """Return all registered semantic relations."""
        return list(self._relations)

    # -- Default dictionary -------------------------------------------------

    @classmethod
    def build_default(cls) -> "ClinicalTerminology":
        """Build the demo semantic relationship dictionary (~50 entries).

        Populates concept-to-semantic-group mappings for all 29 vocabulary
        concepts plus clinical finding targets, and ~50 directed semantic
        relationships.

        Returns
        -------
        ClinicalTerminology
            Populated terminology with concept groups and relationships.
        """
        t = cls()

        # =================================================================
        # Concept → Semantic Group mappings
        # =================================================================

        # Labs → Laboratory Finding
        for lab in [
            "Lactate", "Creatinine", "Bilirubin", "Platelets",
            "WBC", "CRP", "PCT",
        ]:
            t.add_concept_group(lab, "Laboratory Finding")

        # Vitals → Clinical Finding
        for vital in [
            "MAP", "HeartRate", "RespiratoryRate",
            "Temperature", "SpO2", "GCS",
        ]:
            t.add_concept_group(vital, "Clinical Finding")

        # Medications → Pharmacologic Substance
        for med in ["Norepinephrine", "Vasopressin", "Antibiotics"]:
            t.add_concept_group(med, "Pharmacologic Substance")

        # Diagnoses → Diagnostic Concept
        for dx in ["Sepsis", "Pneumonia", "UTI", "AKI", "ARDS"]:
            t.add_concept_group(dx, "Diagnostic Concept")

        # Procedures → Procedure
        for proc in ["Intubation", "CentralLine", "Dialysis"]:
            t.add_concept_group(proc, "Procedure")

        # Clinical Concepts → Clinical Finding
        for cc in [
            "SepticShock", "OrganDysfunction",
            "HemodynamicInstability", "RenalDysfunction",
            "RespiratoryFailure",
        ]:
            t.add_concept_group(cc, "Clinical Finding")

        # Target clinical findings (not in vocab but used as relation targets)
        for finding in [
            "Tissue Hypoxia", "Renal Function", "Hepatic Function",
            "Coagulation", "Immune Response", "Inflammation",
            "Infection", "Hemodynamic Status", "Oxygenation",
            "Consciousness", "Hypotension", "Shock",
            "Organ Failure", "Airway Management",
            "Vascular Access", "Renal Replacement",
        ]:
            t.add_concept_group(finding, "Organ/Tissue Function")

        # =================================================================
        # Semantic Relationships (~50 entries)
        # =================================================================

        # --- Lab biomarker relationships ---
        t.add_relation("Lactate", "is_biomarker_for", "Tissue Hypoxia")
        t.add_relation("Lactate", "indicates", "SepticShock")
        t.add_relation("Lactate", "associated_with", "OrganDysfunction")

        t.add_relation("Creatinine", "is_biomarker_for", "Renal Function")
        t.add_relation("Creatinine", "indicates", "AKI")
        t.add_relation("Creatinine", "associated_with", "RenalDysfunction")

        t.add_relation("Bilirubin", "is_biomarker_for", "Hepatic Function")
        t.add_relation("Bilirubin", "associated_with", "OrganDysfunction")

        t.add_relation("Platelets", "is_biomarker_for", "Coagulation")
        t.add_relation("Platelets", "associated_with", "Sepsis")

        t.add_relation("WBC", "is_biomarker_for", "Immune Response")
        t.add_relation("WBC", "indicates", "Infection")
        t.add_relation("WBC", "associated_with", "Sepsis")

        t.add_relation("CRP", "is_biomarker_for", "Inflammation")
        t.add_relation("CRP", "indicates", "Infection")

        t.add_relation("PCT", "is_biomarker_for", "Infection")
        t.add_relation("PCT", "indicates", "Sepsis")

        # --- Vital sign relationships ---
        t.add_relation("MAP", "indicates", "Hemodynamic Status")
        t.add_relation("MAP", "associated_with", "HemodynamicInstability")
        t.add_relation("MAP", "associated_with", "SepticShock")

        t.add_relation("HeartRate", "indicates", "Hemodynamic Status")
        t.add_relation("HeartRate", "associated_with", "SepticShock")

        t.add_relation("RespiratoryRate", "indicates", "Oxygenation")
        t.add_relation("RespiratoryRate", "associated_with", "RespiratoryFailure")
        t.add_relation("RespiratoryRate", "associated_with", "ARDS")

        t.add_relation("Temperature", "indicates", "Infection")
        t.add_relation("Temperature", "associated_with", "Sepsis")

        t.add_relation("SpO2", "indicates", "Oxygenation")
        t.add_relation("SpO2", "associated_with", "RespiratoryFailure")

        t.add_relation("GCS", "indicates", "Consciousness")
        t.add_relation("GCS", "associated_with", "OrganDysfunction")

        # --- Medication relationships ---
        t.add_relation("Norepinephrine", "treats", "Hypotension")
        t.add_relation("Norepinephrine", "treats", "SepticShock")
        t.add_relation("Norepinephrine", "associated_with", "HemodynamicInstability")

        t.add_relation("Vasopressin", "treats", "Hypotension")
        t.add_relation("Vasopressin", "treats", "SepticShock")

        t.add_relation("Antibiotics", "treats", "Infection")
        t.add_relation("Antibiotics", "treats", "Sepsis")

        # --- Diagnosis relationships ---
        t.add_relation("Sepsis", "causes", "OrganDysfunction")
        t.add_relation("Sepsis", "causes", "SepticShock")

        t.add_relation("Pneumonia", "causes", "RespiratoryFailure")
        t.add_relation("Pneumonia", "associated_with", "Sepsis")

        t.add_relation("UTI", "associated_with", "Sepsis")

        t.add_relation("AKI", "complication_of", "Sepsis")
        t.add_relation("AKI", "associated_with", "RenalDysfunction")

        t.add_relation("ARDS", "causes", "RespiratoryFailure")
        t.add_relation("ARDS", "complication_of", "Sepsis")

        # --- Procedure relationships ---
        t.add_relation("Intubation", "treats", "RespiratoryFailure")
        t.add_relation("Intubation", "associated_with", "Airway Management")

        t.add_relation("CentralLine", "associated_with", "Vascular Access")
        t.add_relation("CentralLine", "associated_with", "Norepinephrine")

        t.add_relation("Dialysis", "treats", "RenalDysfunction")
        t.add_relation("Dialysis", "associated_with", "Renal Replacement")

        return t

