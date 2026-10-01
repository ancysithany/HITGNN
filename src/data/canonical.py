"""
src.data.canonical — Canonical Data Structures for downstream models.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ClinicalEvent:
    """A single observed clinical event in an ICU patient's timeline.

    Attributes
    ----------
    patient_id : str
        Unique patient identifier.
    timestamp : float
        Hours relative to t0 (negative = before t0, 0 = at t0).
    event_type : str
        One of: LAB, VITAL, MEDICATION, PROCEDURE, DIAGNOSIS, CLINICAL_CONCEPT.
    concept_id : int
        Canonical concept ID from ConceptVocabulary.
    concept_name : str
        Human-readable concept name.
    value : float or None
        Numerical value for measurable concepts.
    unit : str or None
        Measurement unit.
    source : str or None
        Origin source dataset (e.g., "synthetic", "mimic").
    """
    patient_id: str
    timestamp: float
    event_type: str
    concept_id: int
    concept_name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None


@dataclass
class PatientRecord:
    """Canonical pre-t0 record for a single ICU patient.

    Attributes
    ----------
    patient_id : str
    age : float
    sex : str
    icu_type : str
    sofa_score : float
    comorbidities : Dict[str, int]
    t0 : float
    mortality : Optional[int]
    split : Optional[str]
    events : List[ClinicalEvent]
    """
    patient_id: str
    age: float
    sex: str
    icu_type: str
    sofa_score: float
    comorbidities: Dict[str, int]
    t0: float
    mortality: Optional[int] = None
    split: Optional[str] = None
    events: List[ClinicalEvent] = field(default_factory=list)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "age": self.age,
            "sex": self.sex,
            "icu_type": self.icu_type,
            "sofa_score": self.sofa_score,
            "comorbidities": self.comorbidities
        }


@dataclass
class CanonicalCohort:
    """A collection of PatientRecords representing a full cohort."""
    patients: List[PatientRecord]
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]

    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        for p in self.patients:
            if p.patient_id == patient_id:
                return p
        return None

    def get_events(self, patient_id: str) -> List[ClinicalEvent]:
        p = self.get_patient(patient_id)
        if p:
            return p.events
        return []

    def get_patient_ids(self) -> List[str]:
        return [p.patient_id for p in self.patients]

    def get_split_ids(self, split: str) -> List[str]:
        if split == "train":
            return self.train_ids
        elif split == "val":
            return self.val_ids
        elif split == "test":
            return self.test_ids
        return []

    def validate(self):
        # Basic validation
        all_split_ids = set(self.train_ids + self.val_ids + self.test_ids)
        all_patient_ids = set(self.get_patient_ids())
        assert all_split_ids.issubset(all_patient_ids), "Some split IDs are not in the cohort"
