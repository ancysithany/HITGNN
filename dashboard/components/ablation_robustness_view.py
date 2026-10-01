import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.utils.loaders import load_ablation_results, load_robustness_results
from dashboard.utils.context import DashboardContext

def apply_dark_theme(fig):
    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    )
    return fig

def render_ablations(ctx: DashboardContext):
    st.title("Ablations & Robustness")
    
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Structural Ablations", "Missingness Robustness"])
    
    with tab1:
        st.markdown("### Phase 9 Structural Ablations")
        df_abl = load_ablation_results()
        
        if df_abl is not None:
            metric = st.selectbox("Select Metric", options=["test_auroc", "test_auprc", "test_f1", "test_brier"])
            
            if metric in df_abl.columns:
                df_sorted = df_abl.sort_values(by=metric, ascending=False)
                fig = px.bar(df_sorted, x=metric, y="variant", orientation='h', title=f"Ablation Performance ({metric.upper()})", color_discrete_sequence=["#6C8CFF"])
                fig = apply_dark_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Metric '{metric}' not found in results.")
        else:
            st.warning("Ablation results not found.")
            
    with tab2:
        st.markdown("### Missingness Robustness")
        st.info("Evaluating model sensitivity to induced missingness across three scenarios: MCAR (Missing Completely At Random), MAR (Missing At Random), and INFORMATIVE missingness.")
        
        df_rob = load_robustness_results()
        if df_rob is not None:
            scenario = st.selectbox("Select Scenario", options=["MCAR", "MAR", "INFORMATIVE"])
            metric_rob = st.selectbox("Select Metric ", options=["auroc", "auprc", "f1", "brier_score"], key="rob_metric")
            
            if "scenario" in df_rob.columns and metric_rob in df_rob.columns:
                df_scenario = df_rob[df_rob["scenario"] == scenario]
                if not df_scenario.empty:
                    fig2 = px.line(df_scenario, x="rate", y=metric_rob, markers=True, title=f"{scenario} Robustness ({metric_rob.upper()})", color_discrete_sequence=["#FB7185"])
                    fig2 = apply_dark_theme(fig2)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info(f"No data for scenario {scenario}")
            else:
                st.dataframe(df_rob, use_container_width=True)
        else:
            st.warning("Robustness results not found.")

    st.markdown('</div>', unsafe_allow_html=True)
