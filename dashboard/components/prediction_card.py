import streamlit as st
import torch
import plotly.graph_objects as go
from torch_geometric.loader import DataLoader
from dashboard.utils.context import DashboardContext
from src.data.canonical import PatientRecord
from dashboard.components.patient_summary import render_patient_summary

def render_gauge(prob, threshold):
    # Dark modern theme gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "%", "valueformat": ".1f", "font": {"color": "#F8FAFC"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#FB7185" if prob >= threshold else "#4ADE80"},
            'bgcolor': "#111827",
            'borderwidth': 2,
            'bordercolor': "#111827",
            'steps': [
                {'range': [0, threshold * 100], 'color': "rgba(255,255,255,0.05)"},
                {'range': [threshold * 100, 100], 'color': "rgba(255,255,255,0.1)"}
            ],
            'threshold': {
                'line': {'color': "#FACC15", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=30, b=20), 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def get_prediction(ctx: DashboardContext, patient: PatientRecord, is_phase8: bool):
    idx = ctx.test_dataset.patient_records.index(patient)
    data = ctx.test_dataset[idx]
    
    batch = next(iter(DataLoader([data], batch_size=1))).to(ctx.device)
    
    model = ctx.phase8_model if is_phase8 else ctx.phase7_model
    threshold = ctx.phase8_threshold if is_phase8 else ctx.phase7_threshold
    
    with torch.no_grad():
        if is_phase8:
            logit = model(batch, ctx.reference_embeddings, ctx.reference_metadata, ctx.reference_ids).item()
        else:
            logit = model(batch).item()
            
    prob = torch.sigmoid(torch.tensor(logit)).item()
    if threshold is not None:
        pred = 1 if prob >= threshold else 0
    else:
        pred = "Unavailable"
    return prob, logit, pred, threshold

def render_prediction_column(title, prob, logit, pred, threshold):
    st.markdown(f"#### {title}")
    
    if threshold is None:
        st.warning("Model Threshold Unavailable")
        return
        
    fig = render_gauge(prob, threshold)
    st.plotly_chart(fig, use_container_width=True)
    
    # Text Metrics
    class_str = "POSITIVE" if pred == 1 else "NEGATIVE"
    class_color = "#FB7185" if pred == 1 else "#4ADE80"
    
    st.markdown(f"""
    <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #94A3B8;'>Predicted class</span>
            <strong style='color: {class_color};'>{class_str}</strong>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #94A3B8;'>Validation threshold</span>
            <strong>{threshold * 100:.1f}%</strong>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #94A3B8;'>Raw logit</span>
            <strong>{logit:.4f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_prediction(ctx: DashboardContext, patient: PatientRecord, selected_model: str):
    st.title("Patient Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### PATIENT CONTEXT")
        render_patient_summary(patient)
        
    with col2:
        st.markdown("### MODEL PREDICTION")
        st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
        
        prob7, logit7, pred7, thresh7 = get_prediction(ctx, patient, is_phase8=False)
        prob8, logit8, pred8, thresh8 = get_prediction(ctx, patient, is_phase8=True)
        
        c1, c2 = st.columns(2)
        with c1:
            render_prediction_column("Patient-only HITGNN", prob7, logit7, pred7, thresh7)
        with c2:
            render_prediction_column("Full HITGNN", prob8, logit8, pred8, thresh8)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        delta_prob = (prob8 - prob7) * 100
        color = "#FB7185" if delta_prob > 0 else "#4ADE80"
        sign = "+" if delta_prob > 0 else ""
        st.markdown(f"""
        <div style='text-align: center;'>
            <span style='color: #94A3B8; font-size: 0.9em; text-transform: uppercase;'>Δ Probability (Phase 8 - Phase 7)</span><br>
            <strong style='color: {color}; font-size: 1.5rem;'>{sign}{delta_prob:.1f}%</strong>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
