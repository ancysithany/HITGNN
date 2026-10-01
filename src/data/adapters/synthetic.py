"""
src.data.adapters.synthetic — Adapter for the static synthetic dataset.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.data.adapters.base import BaseDatasetAdapter
from src.data.canonical import CanonicalCohort, PatientRecord, ClinicalEvent


class SyntheticDatasetAdapter(BaseDatasetAdapter):
    """Adapter for loading the pre-generated static synthetic cohort.
    
    This does NOT generate the cohort. It strictly loads the artifacts
    from the specified synthetic data directory.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.patients_file = self.root_dir / "patients.csv"
        self.events_file = self.root_dir / "events.csv"
        self.splits_file = self.root_dir / "splits.json"
        
        # Internal caches
        self._patients: List[PatientRecord] = []
        self._events: List[ClinicalEvent] = []
        self._splits: Dict[str, List[str]] = {}
        
    def prepare(self) -> CanonicalCohort:
        """Process synthetic CSVs and return the canonical cohort."""
        self.validate()
        
        self._events = self.load_events()
        self._patients = self.load_patients()
        
        with open(self.splits_file, "r") as f:
            self._splits = json.load(f)
            
        # Group events by patient to populate PatientRecord
        events_by_patient = {}
        for ev in self._events:
            events_by_patient.setdefault(ev.patient_id, []).append(ev)
            
        for p in self._patients:
            p.events = events_by_patient.get(p.patient_id, [])
            # Assign split based on loaded split dict
            if p.patient_id in self._splits.get("train", []):
                p.split = "train"
            elif p.patient_id in self._splits.get("val", []):
                p.split = "val"
            elif p.patient_id in self._splits.get("test", []):
                p.split = "test"
                
        # Maintain identical deterministic splits without regenerating
        return CanonicalCohort(
            patients=self._patients,
            train_ids=self._splits.get("train", []),
            val_ids=self._splits.get("val", []),
            test_ids=self._splits.get("test", []),
        )

    def load_patients(self) -> List[PatientRecord]:
        df = pd.read_csv(self.patients_file)
        
        patients = []
        for _, row in df.iterrows():
            comorbidities = {
                "diabetes": int(row["diabetes"]),
                "CKD": int(row["CKD"]),
                "liver_disease": int(row["liver_disease"]),
                "heart_failure": int(row["heart_failure"]),
                "COPD": int(row["COPD"])
            }
            
            p = PatientRecord(
                patient_id=str(row["patient_id"]),
                age=float(row["age"]),
                sex=str(row["sex"]),
                icu_type=str(row["icu_type"]),
                sofa_score=float(row["sofa_score"]),
                comorbidities=comorbidities,
                t0=0.0,  # Pre-t0 convention uses t0=0.0
                mortality=int(row["mortality"])
            )
            patients.append(p)
        return patients

    def load_events(self) -> List[ClinicalEvent]:
        df = pd.read_csv(self.events_file)
        events = []
        for _, row in df.iterrows():
            val = row["value"]
            unit = row["unit"]
            e = ClinicalEvent(
                patient_id=str(row["patient_id"]),
                timestamp=float(row["timestamp"]),
                event_type=str(row["event_type"]),
                concept_id=int(row["concept_id"]),
                concept_name=str(row["concept_name"]),
                value=float(val) if pd.notna(val) else None,
                unit=str(unit) if pd.notna(unit) else None,
                source="synthetic"
            )
            events.append(e)
        return events

    def validate(self) -> None:
        if not self.patients_file.exists():
            raise FileNotFoundError(f"Missing {self.patients_file}")
        if not self.events_file.exists():
            raise FileNotFoundError(f"Missing {self.events_file}")
        if not self.splits_file.exists():
            raise FileNotFoundError(f"Missing {self.splits_file}")
