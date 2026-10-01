"""
src.data.dataset — PyTorch Geometric Dataset for HITGNN.

Wraps the graph builder output into a standard PyTorch Dataset returning PyG Data objects:
    - Patient-level train/val/test split (70/15/15)
    - Temporal cut enforcement (events <= t0 handled by builder)
    - Separates patient metadata (age, SOFA, ICU type, comorbidities) into fixed tensor
    - Attaches Missingness State explicitly
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.canonical import PatientRecord
from src.preprocessing.temporal_binner import TemporalBinner, BiomarkerState
from src.preprocessing.normalizer import FeatureNormalizer
from src.preprocessing.event_normalizer import EventNormalizer
from src.graph.graph_builder import PatientGraphBuilder


class HITGNNPatientDataset(Dataset):
    """Dataset containing per-patient temporal event graphs and missingness state.

    Returns: class:`torch_geometric.data.Data` containing:
        - Graph topology & features (node_type, concept_id, edge_index, etc.)
        - `patient_meta`: FloatTensor [10] with explicit ordered features.
        - `missingness_input`: FloatTensor [K, B, 4] for the missingness encoder.
        - Component raw missingness tensors: biomarker_values, biomarker_masks, etc [K, B].
        - `y`: target label [1].
    """

    def __init__(
        self,
        patient_records: List[PatientRecord],
        graph_builder: PatientGraphBuilder,
        temporal_binner: TemporalBinner,
        biomarker_normalizer: FeatureNormalizer,
        meta_normalizer: Any,
        biomarker_names: List[str],
        split_ids: List[str],
        masking_config: dict = None,
    ) -> None:
        self.patient_records = [p for p in patient_records if p.patient_id in set(split_ids)]
        self.graph_builder = graph_builder
        self.temporal_binner = temporal_binner
        self.biomarker_normalizer = biomarker_normalizer
        self.meta_normalizer = meta_normalizer
        self.biomarker_names = biomarker_names
        self.masking_config = masking_config
        
        # Test aliases
        self.bio_norm = biomarker_normalizer
        self.meta_norm = meta_normalizer

        # Metadata mapping utilities
        self.icu_types = ["MICU", "SICU", "CCU"]

    def _apply_missingness_masking(self, p: PatientRecord) -> List[Any]:
        if not self.masking_config:
            return p.events
            
        scenario = self.masking_config.get("scenario")
        if scenario == "MCAR":
            rate = self.masking_config.get("rate", 0.0)
            rng = np.random.RandomState(self.masking_config.get("seed", 42) + hash(p.patient_id) % 1000000)
            masked_events = []
            for e in p.events:
                if e.event_type == "LAB" and e.concept_name in self.biomarker_names and e.value is not None:
                    if rng.rand() < rate:
                        continue # Mask it out
                masked_events.append(e)
            return masked_events
            
        elif scenario == "MAR":
            rate = self.masking_config.get("rate", 0.0)
            rng = np.random.RandomState(self.masking_config.get("seed", 42) + hash(p.patient_id) % 1000000)
            # logit P(mask) = beta0 + beta1 * age + beta2 * sofa
            age_norm, sofa_norm = self.meta_normalizer.transform(p.metadata["age"], p.metadata["sofa_score"])
            # Choose simple coefficients
            beta1 = 0.5
            beta2 = 0.5
            # We want the average masking rate to be ~rate. We'll use a simple threshold or scale beta0
            # For simplicity, if rate > 0:
            if rate > 0:
                # intercept adjusted roughly
                logit = np.log((rate + 1e-6)/(1 - rate + 1e-6)) + beta1 * age_norm + beta2 * sofa_norm
                prob = 1.0 / (1.0 + np.exp(-logit))
            else:
                prob = 0.0
                
            masked_events = []
            for e in p.events:
                if e.event_type == "LAB" and e.concept_name in self.biomarker_names and e.value is not None:
                    if rng.rand() < prob:
                        continue
                masked_events.append(e)
            return masked_events
            
        elif scenario == "INFORMATIVE":
            rate = self.masking_config.get("rate", 0.0)
            rng = np.random.RandomState(self.masking_config.get("seed", 42) + hash(p.patient_id) % 1000000)
            
            _, sofa_norm = self.meta_normalizer.transform(p.metadata["age"], p.metadata["sofa_score"])
            # Lower SOFA -> higher probability for PCT/CRP
            
            # Base probability for other labs to match roughly
            base_prob = rate
            
            # Inflammatory probability
            if rate > 0:
                logit = np.log((rate + 1e-6)/(1 - rate + 1e-6)) - 1.0 * sofa_norm  # lower SOFA means higher logit
                inf_prob = 1.0 / (1.0 + np.exp(-logit))
            else:
                inf_prob = 0.0
                
            masked_events = []
            for e in p.events:
                if e.event_type == "LAB" and e.concept_name in self.biomarker_names and e.value is not None:
                    curr_prob = inf_prob if e.concept_name in ["PCT", "CRP"] else base_prob
                    if rng.rand() < curr_prob:
                        continue
                masked_events.append(e)
            return masked_events

        return p.events

    def __len__(self) -> int:
        return len(self.patient_records)

    def __getitem__(self, idx: int) -> Data:
        p = self.patient_records[idx]

        # 1. Build Phase 4 Event Graph
        data = self.graph_builder.build(p)

        # 2. Build Phase 3 Biomarker State
        # Find window start based on masked events if configured
        binner_events = self._apply_missingness_masking(p)
        lab_times = [e.timestamp for e in binner_events if e.event_type == "LAB" and e.timestamp <= 0.0]
        window_start = min(lab_times) if lab_times else -24.0

        state = self.temporal_binner.build_biomarker_state(
            binner_events, self.biomarker_names,
            window_start=window_start, window_end=0.0,
            normalizer=self.biomarker_normalizer
        )

        # [n_bins, n_biomarkers, 4] -> [6, 7, 4]. Unsqueezed for PyG batching -> [1, 6, 7]
        data.biomarker_values = torch.tensor(state.values, dtype=torch.float32).unsqueeze(0)
        data.biomarker_masks = torch.tensor(state.masks, dtype=torch.float32).unsqueeze(0)
        data.biomarker_delta_times = torch.tensor(state.delta_times, dtype=torch.float32).unsqueeze(0)
        data.biomarker_never_observed = torch.tensor(state.never_observed, dtype=torch.float32).unsqueeze(0)

        data.missingness_input = torch.stack([
            data.biomarker_values,
            data.biomarker_masks,
            data.biomarker_delta_times,
            data.biomarker_never_observed
        ], dim=-1)

        # 3. Patient Metadata Tensor [10]
        # Order: [age_norm, sofa_norm, icu_MICU, icu_SICU, icu_CCU, diabetes, CKD, liver_disease, heart_failure, COPD]
        md = p.metadata
        age_norm, sofa_norm = self.meta_normalizer.transform(md["age"], md["sofa_score"])
        
        icu_oh = [1.0 if md["icu_type"] == icu else 0.0 for icu in self.icu_types]
        
        meta = [
            age_norm,
            sofa_norm,
            icu_oh[0],
            icu_oh[1],
            icu_oh[2],
            float(md["comorbidities"].get("diabetes", 0)),
            float(md["comorbidities"].get("CKD", 0)),
            float(md["comorbidities"].get("liver_disease", 0)),
            float(md["comorbidities"].get("heart_failure", 0)),
            float(md["comorbidities"].get("COPD", 0)),
        ]
        data.patient_meta = torch.tensor(meta, dtype=torch.float32).unsqueeze(0)

        # 4. Target
        data.y = torch.tensor([p.mortality], dtype=torch.long)
        
        # Patient ID metadata string
        data.patient_id = p.patient_id

        return data
