import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from torch_geometric.loader import DataLoader
from dashboard.utils.context import DashboardContext
from dashboard.utils.loaders import load_case_study
from src.data.canonical import PatientRecord
from src.explainability.attention_explainer import extract_hierarchical_attention
from src.explainability.gradient_explainer import compute_gradient_x_input

def apply_dark_theme(fig):
    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=False)
    )
    return fig

def render_explainability(ctx: DashboardContext, patient: PatientRecord, selected_model: str):
    st.title("Explainability")
    
    if "Phase 8" not in selected_model:
        st.warning("Full Explainability requires the Full HITGNN (Phase 8) model.")
        return
        
    idx = ctx.test_dataset.patient_records.index(patient)
    data = ctx.test_dataset[idx]
    
    if ctx.phase8_model is None or ctx.reference_embeddings is None:
        st.error("Model or Reference Bank unavailable.")
        return
        
    batch = next(iter(DataLoader([data], batch_size=1))).to(ctx.device)
    
    with st.spinner("Extracting Attention and Gradients..."):
        attn_res = extract_hierarchical_attention(patient, batch, ctx.phase8_model, ctx.reference_embeddings, ctx.reference_metadata, ctx.reference_ids)
        grad_res = compute_gradient_x_input(batch, ctx.phase8_model, ctx.reference_embeddings, ctx.reference_metadata, ctx.reference_ids)
        
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Attention", "Gradient × Input", "SHAP", "Temporal Importance"])
    
    with tab1:
        st.markdown("### Event-Type Attention")
        type_imp = attn_res["type_importance"]
        df_type = pd.Series(type_imp).reset_index()
        df_type.columns = ["Event Type", "Attention Weight"]
        fig = px.bar(df_type, x="Attention Weight", y="Event Type", orientation="h", color_discrete_sequence=["#6C8CFF"])
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Top 10 Attention-Associated Events")
        top_events = attn_res["node_importance"][:10]
        df_events = pd.DataFrame(top_events)
        st.dataframe(df_events, use_container_width=True)
        
    with tab2:
        st.markdown("### Gradient × Input Attribution")
        st.info("Gradient × input measures local model sensitivity and is not a causal effect.")
        meta_grad = grad_res["metadata_attribution"]
        df_meta = pd.Series(meta_grad).sort_values().reset_index()
        df_meta.columns = ["Feature", "Attribution"]
        fig2 = px.bar(df_meta, x="Attribution", y="Feature", orientation="h", color="Attribution", color_continuous_scale="RdBu_r")
        fig2 = apply_dark_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab3:
        st.markdown("### SHAP Explanation")
        case_data = load_case_study(patient.patient_id)
        
        if case_data:
            st.success("Loaded precomputed SHAP explanation for this case study.")
            shap_vals = case_data["shap_values"]
            df_shap = pd.DataFrame([
                {"Feature": k, "SHAP Value": v["shap_value"]}
                for k, v in shap_vals.items()
            ]).sort_values("SHAP Value")
            
            fig3 = px.bar(
                df_shap, 
                x="SHAP Value", 
                y="Feature", 
                orientation="h", 
                title="Metadata SHAP (positive increases logit)",
                color="SHAP Value",
                color_continuous_scale="RdBu_r"
            )
            fig3 = apply_dark_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("SHAP explanation has not been precomputed for this patient.")
            st.markdown("Please select one of the four precomputed Case Study patients from the sidebar shortcuts to view SHAP.")
                    
    with tab4:
        st.markdown("### Temporal Importance")
        temp_imp = attn_res["temporal_importance"]
        df_temp = pd.Series(temp_imp).reset_index()
        df_temp.columns = ["Time Window", "Attention Weight"]
        fig4 = px.bar(df_temp, x="Time Window", y="Attention Weight", color_discrete_sequence=["#C084FC"])
        fig4 = apply_dark_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
