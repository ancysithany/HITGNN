import streamlit as st
import torch
import json
import os
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config
from src.data.factory import create_dataset_adapter
from src.preprocessing.pipeline import build_hitgnn_datasets
from src.models.hitgnn import PatientMortalityModel, PopulationHITGNN

@st.cache_data
def get_config():
    return load_config("configs/default.yaml")

@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_data
def get_cohort():
    config = get_config()
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    return cohort

@st.cache_resource
def get_datasets_and_vocab():
    cohort = get_cohort()
    config = get_config()
    train_dataset, val_dataset, test_dataset, vocab = build_hitgnn_datasets(cohort, config)
    return train_dataset, val_dataset, test_dataset, vocab

@st.cache_resource
def load_phase7_model():
    config = get_config()
    device = get_device()
    _, _, _, vocab = get_datasets_and_vocab()
    
    ckpt_path = "outputs/checkpoints/best_patient_model.pt"
    if not os.path.exists(ckpt_path):
        return None
        
    model = PatientMortalityModel(config, vocab.size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

@st.cache_resource
def load_phase8_model():
    config = get_config()
    device = get_device()
    _, _, _, vocab = get_datasets_and_vocab()
    
    ckpt_path = "outputs/checkpoints/best_population_model.pt"
    if not os.path.exists(ckpt_path):
        return None
        
    model = PopulationHITGNN(config, vocab.size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

@st.cache_resource
def get_reference_bank():
    model = load_phase8_model()
    if model is None:
        return None, None, None
        
    device = get_device()
    train_dataset, _, _, _ = get_datasets_and_vocab()
    
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    
    try:
        train_batch = next(iter(train_loader)).to(device)
        with torch.no_grad():
            out = model.encoder(train_batch, return_attention=False)
            ref_emb = out.fused_embedding
            ref_meta = train_batch.patient_meta
            ref_ids = train_batch.patient_id
        return ref_emb, ref_meta, ref_ids
    except Exception as e:
        st.error(f"Failed to build reference bank: {e}")
        return None, None, None

@st.cache_data
def get_phase7_threshold():
    try:
        with open("outputs/metrics/patient_model_metrics.json", "r") as f:
            data = json.load(f)
            return data.get("threshold", None)
    except:
        return None

@st.cache_data
def get_phase8_threshold():
    return 0.01

@st.cache_data
def get_test_patient_options():
    _, _, test_dataset, _ = get_datasets_and_vocab()
    return [p.patient_id for p in test_dataset.patient_records]

@st.cache_data
def load_case_study(patient_id):
    file_path = f"outputs/explanations/{patient_id}.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    res = {}
    try:
        with open("outputs/metrics/patient_model_metrics.json", "r") as f:
            res["phase7"] = json.load(f)
    except:
        res["phase7"] = None
        
    try:
        with open("outputs/metrics/population_model_metrics.json", "r") as f:
            res["phase8"] = json.load(f)
    except:
        res["phase8"] = None
        
    return res

@st.cache_data
def load_ablation_results():
    if os.path.exists("outputs/metrics/ablation_results.csv"):
        return pd.read_csv("outputs/metrics/ablation_results.csv")
    return None

@st.cache_data
def load_robustness_results():
    if os.path.exists("outputs/metrics/missingness_robustness_results.csv"):
        return pd.read_csv("outputs/metrics/missingness_robustness_results.csv")
    return None

@st.cache_data
def get_case_study_mapping():
    return {
        "highest_score_positive": "P0250",
        "lowest_score_positive": "P0133",
        "highest_score_negative": "P0283",
        "lowest_score_negative": "P0362",
    }
