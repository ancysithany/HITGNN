import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.utils.context import DashboardContext
from src.data.canonical import PatientRecord

def render_missingness(ctx: DashboardContext, patient: PatientRecord, selected_model: str):
    st.title("Biomarker Missingness")
    
    idx = ctx.test_dataset.patient_records.index(patient)
    data = ctx.test_dataset[idx]
    
    biomarkers = ["PCT", "CRP", "Lactate", "WBC", "Platelets", "Creatinine", "Bilirubin"]
    windows = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-96h"]
    
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    st.markdown("### Missingness Tensors")
    tab1, tab2, tab3, tab4 = st.tabs(["Observation Mask", "Values", "Delta Time", "Never Observed"])
    
    # Missingness is pre-calculated by temporal_binner.
    # Tensors are [1, 6, 7] shaped typically.
    # data.missingness_input is [K, B, 4]. Let's extract from the data explicitly.
    mask = data.biomarker_masks.squeeze(0).numpy()
    vals = data.biomarker_values.squeeze(0).numpy()
    delta = data.biomarker_delta_times.squeeze(0).numpy()
    never = data.biomarker_never_observed.squeeze(0).numpy()
    
    # We transpose them so Biomarkers are Y and Windows are X as requested.
    mask = mask.T
    vals = vals.T
    delta = delta.T
    
    # For Never Observed, it's just across all time. 
    never_1d = never[0, :]
    
    def apply_dark_theme(fig):
        fig.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        return fig
        
    with tab1:
        st.markdown("**Observed Mask**")
        st.info("1 = Observed, 0 = Missing in this temporal window.")
        fig = px.imshow(mask, x=windows, y=biomarkers, aspect="auto", color_continuous_scale="Blues")
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.markdown("**Normalized Value × Observed Mask**")
        st.info("Biological measurement values (mean-imputed to 0 if missing).")
        fig = px.imshow(vals, x=windows, y=biomarkers, aspect="auto", color_continuous_scale="Viridis")
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.markdown("**Elapsed Time Since Last Observation**")
        st.info("Uses the log(1 + delta_t) transformation as evaluated by the model.")
        fig = px.imshow(delta, x=windows, y=biomarkers, aspect="auto", color_continuous_scale="YlOrRd")
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab4:
        st.markdown("**Never Observed Indicator**")
        st.info("1 = Never observed across entire history.")
        df_never = pd.DataFrame({"Biomarker": biomarkers, "Never Observed": never_1d})
        fig = px.bar(df_never, x="Biomarker", y="Never Observed", color="Never Observed", color_continuous_scale="Reds")
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
