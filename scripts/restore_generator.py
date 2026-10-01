import textwrap
content = """
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.vocab import ConceptVocabulary
from src.data.canonical import ClinicalEvent, PatientRecord

_LAB_PARAMS: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "Lactate":    (1.2,  0.4,  3.0,  2.0,  0.3,  20.0),
    "Creatinine": (0.9,  0.2,  2.0,  1.5,  0.3,  12.0),
    "Bilirubin":  (0.6,  0.3,  3.0,  2.0,  0.1,  30.0),
    "Platelets":  (250., 60.,  -150., 50.,  5.0,  600.),
    "WBC":        (8.0,  2.0,  8.0,  5.0,  0.5,  50.0),
    "CRP":        (5.0,  3.0,  80.,  40.,  0.0,  500.),
    "PCT":        (0.05, 0.02, 5.0,  4.0,  0.01, 100.),
}

_VITAL_PARAMS: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "MAP":             (82., 8.,  -20., 10., 30.,  140.),
    "HeartRate":       (78., 10., 30.,  15., 30.,  200.),
    "RespiratoryRate": (16., 3.,  10.,  5.,  6.,   50.),
    "Temperature":     (37., 0.3, 0.8,  0.5, 34.,  42.),
    "SpO2":            (97., 1.5, -6.,  3.,  60.,  100.),
    "GCS":             (15., 0.5, -5.,  2.,  3.,   15.),
}

_ICU_TYPES = ["MICU", "SICU", "CCU"]

_COMORBIDITIES = ["diabetes", "CKD", "liver_disease", "heart_failure", "COPD"]

_DIAGNOSIS_POOL: List[Tuple[str, float, float]] = [
    ("Sepsis",    0.90, 0.08),
    ("Pneumonia", 0.30, 0.15),
    ("UTI",       0.20, 0.05),
    ("AKI",       0.15, 0.35),
    ("ARDS",      0.05, 0.30),
]

_PROCEDURE_POOL: List[Tuple[str, float, float]] = [
    ("Intubation",  0.05, 0.35),
    ("CentralLine", 0.20, 0.40),
    ("Dialysis",    0.02, 0.20),
]

_CONCEPT_POOL: List[Tuple[str, float, float]] = [
    ("SepticShock",             0.05, 0.45),
    ("OrganDysfunction",        0.15, 0.50),
    ("HemodynamicInstability",  0.10, 0.40),
    ("RenalDysfunction",        0.08, 0.35),
    ("RespiratoryFailure",      0.05, 0.35),
]

class SyntheticCohortGenerator:
    def __init__(self, config: dict, vocab: ConceptVocabulary, seed: int = 42) -> None:
        self.syn_cfg = config.get("synthetic", {})
        self.n_patients = self.syn_cfg.get("n_patients", 500)
        self.target_prevalence = self.syn_cfg.get("mortality_prevalence", 0.20)
        self.min_window = self.syn_cfg.get("min_window_hours", 6)
        self.max_window = self.syn_cfg.get("max_window_hours", 72)
        self.coupling = self.syn_cfg.get("severity_frequency_coupling", 0.5)
        self.vocab = vocab
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate(self) -> List[PatientRecord]:
        patients = []
        for i in range(self.n_patients):
            pid = f"P{i:04d}"
            patient = self._generate_patient(pid)
            patients.append(patient)
        return patients

    def _generate_patient(self, patient_id: str) -> PatientRecord:
        severity = self.rng.beta(2.0, 5.0)
        metadata = self._generate_metadata(severity)
        mortality = self._generate_mortality(severity, metadata)
        window_hours = self.rng.uniform(self.min_window, self.max_window)

        events: List[ClinicalEvent] = []
        events.extend(self._generate_lab_events(patient_id, severity, window_hours))
        events.extend(self._generate_vital_events(patient_id, severity, window_hours))
        events.extend(self._generate_medication_events(patient_id, severity, window_hours))
        events.extend(self._generate_diagnosis_events(patient_id, severity, window_hours))
        events.extend(self._generate_procedure_events(patient_id, severity, window_hours))
        events.extend(self._generate_clinical_concept_events(patient_id, severity, window_hours))

        events.sort(key=lambda e: e.timestamp)
        events = [e for e in events if e.timestamp <= 0.0]

        # Use temporary field or internal passing for severity if needed, but not on canonical structure!
        # Actually, PatientRecord does not have severity anymore.
        # But we need to use it inside generation.
        # It's fine to just return the PatientRecord.
        
        pr = PatientRecord(
            patient_id=patient_id,
            age=float(metadata["age"]),
            sex=metadata["sex"],
            icu_type=metadata["icu_type"],
            sofa_score=float(metadata["sofa_score"]),
            comorbidities=metadata["comorbidities"],
            t0=0.0,
            mortality=mortality,
            events=events
        )
        # Hack to attach severity just for the generator debugging outputs
        pr._generator_latent_severity = severity
        return pr

    def _generate_metadata(self, severity: float) -> Dict[str, Any]:
        age = int(np.clip(self.rng.normal(65 + severity * 8, 12), 18, 95))
        sex = self.rng.choice(["M", "F"], p=[0.55, 0.45])
        icu_type = self.rng.choice(_ICU_TYPES, p=[0.50, 0.30, 0.20])
        sofa_base = severity * 14 + self.rng.normal(0, 2.5)
        sofa_score = int(np.clip(sofa_base, 0, 24))

        comorbidities = {}
        base_probs = [0.25, 0.12, 0.08, 0.15, 0.12]
        sev_bonus = [0.10, 0.15, 0.10, 0.10, 0.05]
        for name, bp, sb in zip(_COMORBIDITIES, base_probs, sev_bonus):
            prob = bp + sb * severity + self.rng.normal(0, 0.05)
            comorbidities[name] = int(self.rng.random() < np.clip(prob, 0, 0.8))

        return {
            "age": age,
            "sex": sex,
            "icu_type": icu_type,
            "sofa_score": sofa_score,
            "comorbidities": comorbidities,
        }

    def _generate_mortality(self, severity: float, metadata: Dict[str, Any]) -> int:
        logit = -3.2 + 5.0 * severity
        logit += (metadata["age"] - 65) / 30.0 * 0.3
        logit += (metadata["sofa_score"] - 6) / 18.0 * 0.5
        n_comorbidities = sum(metadata["comorbidities"].values())
        logit += n_comorbidities * 0.12
        logit += self.rng.normal(0, 1.0)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return int(self.rng.random() < prob)

    def _generate_lab_events(self, pid: str, severity: float, window_hours: float) -> List[ClinicalEvent]:
        events: List[ClinicalEvent] = []
        lab_ordering: List[Tuple[str, float, float, float]] = [
            ("Lactate",    0.70, 1.5, 3.0),
            ("Creatinine", 0.85, 1.0, 1.5),
            ("Bilirubin",  0.50, 0.5, 1.0),
            ("Platelets",  0.80, 0.8, 1.0),
            ("WBC",        0.85, 0.8, 1.0),
            ("CRP",        0.35, 0.3, 0.5),
            ("PCT",        0.15, 0.2, 0.4),
        ]
        coupling = self.coupling
        for concept_name, base_p_order, freq_base, freq_sev_bonus in lab_ordering:
            p_ordered = base_p_order + coupling * severity * 0.2
            p_ordered += self.rng.normal(0, 0.1)
            p_ordered = np.clip(p_ordered, 0.05, 0.95)
            if self.rng.random() > p_ordered:
                continue
            rate_per_24h = freq_base + coupling * severity * freq_sev_bonus
            rate_per_24h += self.rng.exponential(0.5)
            rate_per_24h = max(rate_per_24h, 0.2)
            expected_count = rate_per_24h * (window_hours / 24.0)
            n_measurements = max(1, self.rng.poisson(expected_count))
            timestamps = self._generate_irregular_timestamps(window_hours, n_measurements)
            concept = self.vocab.get_by_name(concept_name)
            params = _LAB_PARAMS[concept_name]
            values = self._generate_lab_trajectory(severity, params, n_measurements)
            for t_stamp, val in zip(timestamps, values):
                events.append(ClinicalEvent(
                    patient_id=pid,
                    timestamp=t_stamp,
                    event_type="LAB",
                    concept_id=concept.concept_id,
                    concept_name=concept_name,
                    value=round(float(val), 2),
                    unit=concept.unit,
                    source="synthetic"
                ))
        return events

    def _generate_vital_events(self, pid: str, severity: float, window_hours: float) -> List[ClinicalEvent]:
        events: List[ClinicalEvent] = []
        vital_freq: List[Tuple[str, float, float]] = [
            ("MAP",             0.8, 0.5),
            ("HeartRate",       0.8, 0.5),
            ("RespiratoryRate", 0.5, 0.3),
            ("Temperature",     0.25, 0.1),
            ("SpO2",            0.8, 0.5),
            ("GCS",             0.15, 0.2),
        ]
        for concept_name, freq_base, freq_bonus in vital_freq:
            rate = freq_base + self.coupling * severity * freq_bonus
            rate += self.rng.exponential(0.1)
            expected_count = rate * window_hours
            n_measurements = max(2, self.rng.poisson(max(expected_count, 2)))
            n_measurements = min(n_measurements, int(window_hours * 4))
            timestamps = self._generate_irregular_timestamps(window_hours, n_measurements, jitter_fraction=0.3)
            concept = self.vocab.get_by_name(concept_name)
            params = _VITAL_PARAMS[concept_name]
            values = self._generate_vital_trajectory(severity, params, n_measurements, concept_name)
            for t_stamp, val in zip(timestamps, values):
                events.append(ClinicalEvent(
                    patient_id=pid,
                    timestamp=t_stamp,
                    event_type="VITAL",
                    concept_id=concept.concept_id,
                    concept_name=concept_name,
                    value=round(float(val), 1),
                    unit=concept.unit,
                    source="synthetic"
                ))
        return events

    def _generate_medication_events(self, pid: str, severity: float, window_hours: float) -> List[ClinicalEvent]:
        events: List[ClinicalEvent] = []
        meds: List[Tuple[str, float, float, bool]] = [
            ("Norepinephrine", 0.05, 0.50, True),
            ("Vasopressin",    0.02, 0.25, True),
            ("Antibiotics",    0.70, 0.20, False),
        ]
        for concept_name, base_p, sev_bonus, has_dose in meds:
            prob = base_p + sev_bonus * severity + self.rng.normal(0, 0.05)
            prob = np.clip(prob, 0.0, 0.95)
            if self.rng.random() > prob:
                continue
            concept = self.vocab.get_by_name(concept_name)
            if concept_name == "Antibiotics":
                n_events = self.rng.randint(1, 4)
            else:
                n_events = max(1, self.rng.poisson(2 + severity * 3))
                n_events = min(n_events, int(window_hours / 2))
            timestamps = self._generate_irregular_timestamps(window_hours, n_events)
            for t_stamp in timestamps:
                val = None
                if has_dose:
                    if concept_name == "Norepinephrine":
                        val = round(float(self.rng.uniform(0.02, 0.5 + severity * 0.5)), 3)
                    elif concept_name == "Vasopressin":
                        val = round(float(self.rng.uniform(0.01, 0.04)), 3)
                events.append(ClinicalEvent(
                    patient_id=pid,
                    timestamp=t_stamp,
                    event_type="MEDICATION",
                    concept_id=concept.concept_id,
                    concept_name=concept_name,
                    value=val,
                    unit=concept.unit,
                    source="synthetic"
                ))
        return events

    def _generate_diagnosis_events(self, pid: str, severity: float, window_hours: float) -> List[ClinicalEvent]:
        events: List[ClinicalEvent] = []
        for concept_name, base_p, sev_bonus in _DIAGNOSIS_POOL:
            prob = base_p + sev_bonus * severity + self.rng.normal(0, 0.05)
            prob = np.clip(prob, 0.0, 0.98)
            if self.rng.random() > prob:
                continue
            concept = self.vocab.get_by_name(concept_name)
            if concept_name == "Sepsis":
                t_stamp = self.rng.uniform(-2.0, 0.0)
            elif concept_name in ("AKI", "ARDS"):
                t_stamp = self.rng.uniform(-window_hours * 0.5, 0.0)
            else:
                t_stamp = self.rng.uniform(-window_hours, -window_hours * 0.5)
            events.append(ClinicalEvent(
                patient_id=pid,
                timestamp=round(float(t_stamp), 2),
                event_type="DIAGNOSIS",
                concept_id=concept.concept_id,
                concept_name=concept_name,
                value=None,
                unit=None,
                source="synthetic"
            ))
        return events

    def _generate_procedure_events(self, pid: str, severity: float, window_hours: float) -> List[ClinicalEvent]:
        events: List[ClinicalEvent] = []
        for concept_name, base_p, sev_bonus in _PROCEDURE_POOL:
            prob = base_p + sev_bonus * severity + self.rng.normal(0, 0.03)
            prob = np.clip(prob, 0.0, 0.90)
            if self.rng.random() > prob:
                continue
            concept = self.vocab.get_by_name(concept_name)
            t_stamp = self.rng.uniform(-window_hours * 0.8, -0.5)
            events.append(ClinicalEvent(
                patient_id=pid,
                timestamp=round(float(t_stamp), 2),
                event_type="PROCEDURE",
                concept_id=concept.concept_id,
                concept_name=concept_name,
                value=None,
                unit=None,
                source="synthetic"
            ))
        return events

    def _generate_clinical_concept_events(self, pid: str, severity: float, window_hours: float) -> List[ClinicalEvent]:
        events: List[ClinicalEvent] = []
        for concept_name, base_p, sev_bonus in _CONCEPT_POOL:
            prob = base_p + sev_bonus * severity + self.rng.normal(0, 0.05)
            prob = np.clip(prob, 0.0, 0.90)
            if self.rng.random() > prob:
                continue
            concept = self.vocab.get_by_name(concept_name)
            n_occurrences = 1 + self.rng.poisson(severity * 1.5)
            n_occurrences = min(n_occurrences, 4)
            for _ in range(n_occurrences):
                t_stamp = self.rng.uniform(-window_hours * 0.9, 0.0)
                events.append(ClinicalEvent(
                    patient_id=pid,
                    timestamp=round(float(t_stamp), 2),
                    event_type="CLINICAL_CONCEPT",
                    concept_id=concept.concept_id,
                    concept_name=concept_name,
                    value=None,
                    unit=None,
                    source="synthetic"
                ))
        return events

    def _generate_irregular_timestamps(self, window_hours: float, n_points: int, jitter_fraction: float = 0.4) -> List[float]:
        if n_points <= 0: return []
        if n_points == 1: return [round(float(self.rng.uniform(-window_hours, 0.0)), 2)]
        base = np.linspace(-window_hours, 0.0, n_points)
        gap = window_hours / max(n_points - 1, 1)
        jitter = self.rng.normal(0, gap * jitter_fraction, size=n_points)
        timestamps = base + jitter
        timestamps = np.clip(timestamps, -window_hours, 0.0)
        timestamps.sort()
        return [round(float(t), 2) for t in timestamps]

    def _generate_lab_trajectory(self, severity: float, params: Tuple[float, float, float, float, float, float], n_points: int) -> List[float]:
        normal_mean, normal_std, sev_shift, sev_extra_std, clamp_lo, clamp_hi = params
        base = normal_mean + severity * sev_shift
        std = normal_std + severity * sev_extra_std
        values = [self.rng.normal(base, std)]
        for _ in range(n_points - 1):
            drift = self.rng.normal(0, std * 0.3)
            noise = self.rng.normal(0, std * 0.5)
            new_val = values[-1] + drift + noise
            values.append(new_val)
        values = [np.clip(v, clamp_lo, clamp_hi) for v in values]
        return values

    def _generate_vital_trajectory(self, severity: float, params: Tuple[float, float, float, float, float, float], n_points: int, concept_name: str) -> List[float]:
        normal_mean, normal_std, sev_shift, sev_extra_std, clamp_lo, clamp_hi = params
        base = normal_mean + severity * sev_shift
        std = normal_std + severity * sev_extra_std
        values = []
        prev = self.rng.normal(base, std)
        for _ in range(n_points):
            noise = self.rng.normal(0, std * 0.4)
            mean_revert = 0.1 * (base - prev)
            prev = prev + mean_revert + noise
            prev = np.clip(prev, clamp_lo, clamp_hi)
            if concept_name == "GCS":
                values.append(float(int(round(prev))))
            else:
                values.append(prev)
        return values


def save_cohort(patients: List[PatientRecord], output_dir: str = "data/synthetic") -> Dict[str, str]:
    import os
    os.makedirs(output_dir, exist_ok=True)
    patient_rows = []
    debug_rows = []
    for p in patients:
        row = {
            "patient_id": p.patient_id,
            "age": p.age,
            "sex": p.sex,
            "icu_type": p.icu_type,
            "sofa_score": p.sofa_score,
            "mortality": p.mortality,
            "n_events": len(p.events),
        }
        for comorb in _COMORBIDITIES:
            row[comorb] = p.comorbidities.get(comorb, 0)
        patient_rows.append(row)
        debug_rows.append({
            "patient_id": p.patient_id,
            "_generator_latent_severity": round(getattr(p, "_generator_latent_severity", 0.0), 6),
        })
    patients_df = pd.DataFrame(patient_rows)
    patients_path = os.path.join(output_dir, "patients.csv")
    patients_df.to_csv(patients_path, index=False)

    debug_df = pd.DataFrame(debug_rows)
    debug_path = os.path.join(output_dir, "_generator_debug.csv")
    debug_df.to_csv(debug_path, index=False)

    event_rows = []
    for p in patients:
        for e in p.events:
            event_rows.append({
                "patient_id": e.patient_id,
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "concept_id": e.concept_id,
                "concept_name": e.concept_name,
                "value": e.value,
                "unit": e.unit,
                "source": e.source
            })
    events_df = pd.DataFrame(event_rows)
    events_path = os.path.join(output_dir, "events.csv")
    events_df.to_csv(events_path, index=False)

    return {
        "patients": patients_path,
        "events": events_path,
    }

def patient_level_split(patient_ids: List[str], ratios: Dict[str, float], seed: int = 42, labels: Optional[List[int]] = None) -> Dict[str, List[str]]:
    rng = np.random.RandomState(seed)
    if labels is None:
        ids = list(patient_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        return {
            "train": ids[:n_train],
            "val": ids[n_train:n_train+n_val],
            "test": ids[n_train+n_val:]
        }
    else:
        # Stratified split using numpy
        ids = np.array(patient_ids)
        labels = np.array(labels)
        
        # Determine target counts based on actual dataset size, not 500
        n_total = len(ids)
        
        # Use exact target counts from actual dataset size to match legacy exactly
        target_train = int(n_total * ratios["train"])
        target_val = int(n_total * ratios["val"])
        target_test = n_total - target_train - target_val
        
        train_ids = []
        val_ids = []
        test_ids = []
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            label_ids = ids[label_mask]
            rng.shuffle(label_ids)
            
            # Use same proportioning as target dataset size
            n_label = len(label_ids)
            n_l_train = int(n_label * ratios["train"])
            n_l_val = int(n_label * ratios["val"])
            
            train_ids.extend(label_ids[:n_l_train])
            val_ids.extend(label_ids[n_l_train:n_l_train+n_l_val])
            test_ids.extend(label_ids[n_l_train+n_l_val:])
            
        return {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }
"""
with open("src/data/synthetic_generator.py", "w") as f:
    f.write(content)
