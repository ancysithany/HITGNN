import streamlit as st
from src.data.canonical import PatientRecord

def render_patient_summary(patient: PatientRecord):
    # Top metrics card
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AGE", f"{patient.age:.1f}")
    c2.metric("SEX", patient.sex)
    c3.metric("ICU TYPE", patient.icu_type)
    c4.metric("SOFA", f"{patient.sofa_score}")
    
    # Comorbidities
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    st.markdown("<p style='text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 0.5rem;'>Comorbidities</p>", unsafe_allow_html=True)
    
    comorbidities = []
    if isinstance(patient.comorbidities, dict):
        for k, v in patient.comorbidities.items():
            if v:
                # Format name nicely
                name = k.replace("_", " ").title()
                if k.lower() == "ckd": name = "CKD"
                if k.lower() == "copd": name = "COPD"
                comorbidities.append(name)
    elif isinstance(patient.comorbidities, (list, tuple, set)):
        for k in patient.comorbidities:
            name = str(k).replace("_", " ").title()
            if str(k).lower() == "ckd": name = "CKD"
            if str(k).lower() == "copd": name = "COPD"
            comorbidities.append(name)
            
    if comorbidities:
        badges = " &bull; ".join([f"<strong>{c}</strong>" for c in comorbidities])
        st.markdown(f"<p>{badges}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p>None recorded</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Reference Outcome
    st.markdown('<div class="hitgnn-card" style="border-left: 4px solid #94A3B8;">', unsafe_allow_html=True)
    st.markdown("<p style='text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 0.5rem;'>Reference outcome — not used as model input</p>", unsafe_allow_html=True)
    gt = "Non-survivor (Positive)" if patient.mortality == 1 else "Survivor (Negative)"
    st.markdown(f"<p style='font-size: 1.1rem; font-weight: 600; color: #F8FAFC;'>{gt}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
