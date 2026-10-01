"""
src.data.adapters.mimic — Skeleton for future MIMIC-IV extraction.
"""
from pathlib import Path
from typing import List

from src.data.adapters.base import BaseDatasetAdapter
from src.data.canonical import CanonicalCohort, PatientRecord, ClinicalEvent

class MIMICDatasetAdapter(BaseDatasetAdapter):
    """Adapter for extracting a cohort from MIMIC-IV.
    
    Currently a skeleton indicating the architectural boundary.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        # Expected MIMIC-IV CSVs or DB connection params would be here
        
    def prepare(self) -> CanonicalCohort:
        raise NotImplementedError("MIMIC-IV extraction is not implemented in Phase 8.5")

    def load_patients(self) -> List[PatientRecord]:
        raise NotImplementedError("MIMIC-IV extraction is not implemented in Phase 8.5")

    def load_events(self) -> List[ClinicalEvent]:
        raise NotImplementedError("MIMIC-IV extraction is not implemented in Phase 8.5")
        
    def validate(self) -> None:
        raise NotImplementedError("MIMIC-IV extraction is not implemented in Phase 8.5")
