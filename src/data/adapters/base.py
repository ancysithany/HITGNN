"""
src.data.adapters.base — Base interface for dataset adapters.
"""
from abc import ABC, abstractmethod
from typing import List

from src.data.canonical import CanonicalCohort, PatientRecord, ClinicalEvent

class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset ingestion adapters.
    
    Adapters translate raw dataset sources into CanonicalCohort objects.
    """
    
    @abstractmethod
    def prepare(self) -> CanonicalCohort:
        """Process raw data and return the canonical cohort."""
        pass
    
    @abstractmethod
    def load_patients(self) -> List[PatientRecord]:
        """Load and return all canonical patient records."""
        pass
    
    @abstractmethod
    def load_events(self) -> List[ClinicalEvent]:
        """Load and return all canonical events."""
        pass
    
    @abstractmethod
    def validate(self) -> None:
        """Validate the loaded data."""
        pass
