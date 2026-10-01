import streamlit as st

def render_overview():
    st.title("HITGNN")
    st.subheader("Hierarchical Temporal Event Graph Neural Network for Missingness-Aware Sepsis Mortality Prediction")
    
    st.markdown("---")
    
    st.markdown("""
    ### Architecture Overview
    
    Patient Events
    ↓
    Event Node Encoding
    ↓
    Time-aware Graph Attention
    ↓
    Hierarchical Pooling
    ↓
    Missingness-aware Biomarker Encoding
    ↓
    Patient Representation
    ↓
    Population Graph Context
    ↓
    Mortality Prediction
    """)
    
    st.markdown("---")
    st.markdown("### Cohort Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", "500")
    col2.metric("Train", "349")
    col3.metric("Validation", "74")
    col4.metric("Test", "77")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6, _, _ = st.columns(4)
    col5.metric("Biomarkers", "7")
    col6.metric("Temporal Windows", "6")
