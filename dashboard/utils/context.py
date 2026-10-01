import streamlit as st
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.data.canonical import PatientRecord
from dashboard.utils.loaders import (
    load_phase7_model, 
    load_phase8_model, 
    get_datasets_and_vocab, 
    get_reference_bank,
    get_phase7_threshold,
    get_phase8_threshold,
    get_device
)

@dataclass
class DashboardContext:
    device: Any
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    vocab: Any
    phase7_model: Any
    phase8_model: Any
    reference_embeddings: Any
    reference_metadata: Any
    reference_ids: Any
    phase7_threshold: Optional[float]
    phase8_threshold: float

@st.cache_resource(show_spinner="Loading HITGNN Context...")
def load_dashboard_context() -> DashboardContext:
    train_ds, val_ds, test_ds, vocab = get_datasets_and_vocab()
    m7 = load_phase7_model()
    m8 = load_phase8_model()
    ref_emb, ref_meta, ref_ids = get_reference_bank()
    t7 = get_phase7_threshold()
    t8 = get_phase8_threshold()
    
    return DashboardContext(
        device=get_device(),
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        vocab=vocab,
        phase7_model=m7,
        phase8_model=m8,
        reference_embeddings=ref_emb,
        reference_metadata=ref_meta,
        reference_ids=ref_ids,
        phase7_threshold=t7,
        phase8_threshold=t8
    )

def get_patient_record(context: DashboardContext, patient_id: str) -> Optional[PatientRecord]:
    for p in context.test_dataset.patient_records:
        if p.patient_id == patient_id:
            return p
    return None
