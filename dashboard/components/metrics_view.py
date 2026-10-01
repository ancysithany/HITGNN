import streamlit as st
import pandas as pd
import os
from dashboard.utils.loaders import load_metrics
from dashboard.utils.context import DashboardContext

def render_metrics(ctx: DashboardContext):
    st.title("Model Results")
    
    metrics = load_metrics()
    
    st.markdown("### Test Set Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
        st.markdown("#### Patient-only HITGNN (Phase 7)")
        if metrics.get("phase7"):
            m7 = metrics["phase7"]
            c1, c2 = st.columns(2)
            c1.metric("AUROC", f"{m7.get('auroc', 0):.4f}")
            c2.metric("AUPRC", f"{m7.get('auprc', 0):.4f}")
            c1.metric("F1 Score", f"{m7.get('f1', 0):.4f}")
            c2.metric("Brier Score", f"{m7.get('brier', 0):.4f}")
        else:
            st.warning("Phase 7 metrics not found.")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
        st.markdown("#### Full HITGNN (Phase 8)")
        if metrics.get("phase8"):
            m8 = metrics["phase8"]
            c1, c2 = st.columns(2)
            c1.metric("AUROC", f"{m8.get('auroc', 0):.4f}")
            c2.metric("AUPRC", f"{m8.get('auprc', 0):.4f}")
            c1.metric("F1 Score", f"{m8.get('f1', 0):.4f}")
            c2.metric("Brier Score", f"{m8.get('brier', 0):.4f}")
        else:
            st.warning("Phase 8 metrics not found.")
        st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown("### Evaluation Visualizations")
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["ROC & PR Curves", "Confusion Matrix", "Calibration"])
    
    with tab1:
        if os.path.exists("outputs/figures/population_model_roc_pr.png"):
            st.image("outputs/figures/population_model_roc_pr.png")
        else:
            st.info("ROC/PR curves not found.")
            
    with tab2:
        if os.path.exists("outputs/figures/population_model_cm.png"):
            st.image("outputs/figures/population_model_cm.png")
        else:
            st.info("Confusion matrix not found.")
            
    with tab3:
        if os.path.exists("outputs/figures/population_model_calibration.png"):
            st.image("outputs/figures/population_model_calibration.png")
        else:
            st.info("Calibration curve not found.")
            
    st.markdown('</div>', unsafe_allow_html=True)
