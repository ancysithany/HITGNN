import streamlit as st
from dashboard.utils.context import DashboardContext

def render_sidebar(ctx: DashboardContext):
    
    # --- MODEL ---
    st.sidebar.markdown("### MODEL")
    selected_model = st.sidebar.selectbox(
        "Model Version",
        options=["Full HITGNN (Phase 8)", "Patient-only HITGNN (Phase 7)"],
        label_visibility="collapsed"
    )
    
    # --- PATIENT ---
    st.sidebar.markdown("### PATIENT")
    test_patients = [p.patient_id for p in ctx.test_dataset.patient_records]
    
    if "selected_patient" not in st.session_state:
        st.session_state["selected_patient"] = test_patients[0]
        
    selected_patient = st.sidebar.selectbox(
        "Select Patient",
        options=test_patients,
        index=test_patients.index(st.session_state["selected_patient"]) if st.session_state["selected_patient"] in test_patients else 0,
        label_visibility="collapsed"
    )
    
    if selected_patient != st.session_state["selected_patient"]:
        st.session_state["selected_patient"] = selected_patient
        
    # --- CASE STUDIES ---
    st.sidebar.markdown("### CASE STUDIES")
    # Load case study mapping
    from dashboard.utils.loaders import get_case_study_mapping
    mapping = get_case_study_mapping()
    
    if mapping:
        for case_name, pid in mapping.items():
            if st.sidebar.button(case_name.replace("_", " ").title(), use_container_width=True):
                st.session_state["selected_patient"] = pid
                st.rerun()

    # --- NAVIGATION ---
    st.sidebar.markdown("### NAVIGATION")
    pages = [
        "Overview",
        "Patient Prediction",
        "Clinical Timeline",
        "Event Graph",
        "Missingness",
        "Explainability",
        "Population Context",
        "Model Results",
        "Ablations & Robustness"
    ]
    
    # Instead of radio buttons, we can use a selectbox or segmented control or just radio without label
    page = st.sidebar.radio(
        "Navigation",
        options=pages,
        label_visibility="collapsed"
    )
    
    return page, st.session_state["selected_patient"], selected_model
