import streamlit as st

st.set_page_config(
    page_title="HITGNN Research Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path

# Add project root to path so 'dashboard' and 'src' are discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.utils.context import load_dashboard_context, get_patient_record
from dashboard.components.sidebar import render_sidebar
from dashboard.components.overview import render_overview
from dashboard.components.prediction_card import render_prediction
from dashboard.components.timeline_view import render_timeline
from dashboard.components.graph_view import render_graph
from dashboard.components.missingness_view import render_missingness
from dashboard.components.explainability_view import render_explainability
from dashboard.components.population_view import render_population
from dashboard.components.metrics_view import render_metrics
from dashboard.components.ablation_robustness_view import render_ablations

def main():
    # Load Styles
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Load Context once
    try:
        ctx = load_dashboard_context()
    except Exception as e:
        st.error(f"Failed to load dashboard context: {str(e)}")
        return

    # Render Sidebar and get state
    page, selected_patient, selected_model = render_sidebar(ctx)
    
    # Get structured patient record
    patient = None
    if selected_patient:
        patient = get_patient_record(ctx, selected_patient)
    
    # Header
    st.markdown(
        """
        <div style='margin-bottom: 2rem;'>
            <h1 style='margin-bottom: 0;'>HITGNN <span style='font-size: 0.5em; background-color: #6C8CFF; color: white; padding: 2px 8px; border-radius: 4px; vertical-align: middle;'>RESEARCH DEMO</span></h1>
            <p style='color: #94A3B8; font-size: 1.1em; margin-top: 0;'>Hierarchical Temporal Event Graph Neural Network</p>
            <p style='color: #94A3B8; font-size: 0.9em;'>Missingness-aware sepsis mortality prediction using temporal event graphs.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Error boundary wrapper
    try:
        # Routing
        if page == "Overview":
            render_overview()
        elif page == "Patient Prediction":
            if patient:
                render_prediction(ctx, patient, selected_model)
            else:
                st.warning("Please select a valid patient from the sidebar.")
        elif page == "Clinical Timeline":
            if patient:
                render_timeline(ctx, patient)
            else:
                st.warning("Please select a valid patient from the sidebar.")
        elif page == "Event Graph":
            if patient:
                render_graph(ctx, patient, selected_model)
            else:
                st.warning("Please select a valid patient from the sidebar.")
        elif page == "Missingness":
            if patient:
                render_missingness(ctx, patient, selected_model)
            else:
                st.warning("Please select a valid patient from the sidebar.")
        elif page == "Explainability":
            if patient:
                render_explainability(ctx, patient, selected_model)
            else:
                st.warning("Please select a valid patient from the sidebar.")
        elif page == "Population Context":
            if patient:
                render_population(ctx, patient, selected_model)
            else:
                st.warning("Please select a valid patient from the sidebar.")
        elif page == "Model Results":
            render_metrics(ctx)
        elif page == "Ablations & Robustness":
            render_ablations(ctx)
    except Exception as e:
        st.error("Unable to render this section. See terminal logs for details.")
        import traceback
        traceback.print_exc()
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

    # Footer Disclaimer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #94A3B8; font-size: 0.8em; margin-top: 2rem;'>
            <i>Research demonstration using synthetic patient data.<br>
            Not intended for clinical diagnosis, treatment, or decision-making.</i>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
