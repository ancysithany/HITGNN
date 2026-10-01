"""
src.preprocessing.pipeline — Central shared preprocessing pipeline.
"""
from typing import Dict, Tuple
import numpy as np

from src.data.canonical import CanonicalCohort
from src.data.dataset import HITGNNPatientDataset
from src.graph.graph_builder import PatientGraphBuilder
from src.preprocessing.normalizer import FeatureNormalizer
from src.preprocessing.event_normalizer import EventNormalizer
from src.preprocessing.meta_normalizer import PatientMetaNormalizer
from src.preprocessing.temporal_binner import TemporalBinner
from src.data.vocab import ConceptVocabulary
from src.data.terminology import ClinicalTerminology

def build_hitgnn_datasets(
    cohort: CanonicalCohort,
    config: dict,
    meta_norm_state: dict = None
) -> Tuple[HITGNNPatientDataset, HITGNNPatientDataset, HITGNNPatientDataset, ConceptVocabulary]:
    """Process a canonical cohort and return the 3 splits of HITGNNPatientDataset."""
    
    vocab = ConceptVocabulary.build_default()
    term = ClinicalTerminology.build_default()
    
    biomarkers = config.get("biomarkers", [])
    
    bio_norm = FeatureNormalizer()
    event_norm = EventNormalizer()
    meta_norm = PatientMetaNormalizer()
    
    # We fit normalizers ONLY on the training split!
    train_patients = [p for p in cohort.patients if p.split == "train"]
    
    for bm in biomarkers:
        vals = [e.value for p in train_patients for e in p.events
                if e.event_type == "LAB" and e.concept_name == bm and e.value is not None]
        bio_norm.fit(bm, np.array(vals) if vals else np.array([]))
        
    continuous = biomarkers + ["MAP", "Heart Rate", "Respiratory Rate", "Temperature", "SpO2", "GCS"]
    for c in continuous:
        vals = [e.value for p in train_patients for e in p.events
                if e.concept_name == c and e.value is not None]
        event_norm.fit(c, np.array(vals) if vals else np.array([]))
        
    if meta_norm_state is not None:
        meta_norm.load_state_dict(meta_norm_state)
    else:
        meta_norm.fit(train_patients)
        
    graph_builder = PatientGraphBuilder(vocab, term, event_norm, config.get("graph", {}))
    binner = TemporalBinner(n_bins=config.get("temporal", {}).get("n_bins", 6))
    
    train_dataset = HITGNNPatientDataset(cohort.patients, graph_builder, binner, bio_norm, meta_norm, biomarkers, cohort.train_ids)
    val_dataset = HITGNNPatientDataset(cohort.patients, graph_builder, binner, bio_norm, meta_norm, biomarkers, cohort.val_ids)
    test_dataset = HITGNNPatientDataset(cohort.patients, graph_builder, binner, bio_norm, meta_norm, biomarkers, cohort.test_ids)
    
    return train_dataset, val_dataset, test_dataset, vocab
