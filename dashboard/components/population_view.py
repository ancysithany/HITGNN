import streamlit as st
import pandas as pd
import torch
import streamlit.components.v1 as components
from torch_geometric.loader import DataLoader
from dashboard.utils.context import DashboardContext
from src.data.canonical import PatientRecord
import torch.nn.functional as F

def render_population(ctx: DashboardContext, patient: PatientRecord, selected_model: str):
    st.title("Population Context")
    
    if "Phase 8" not in selected_model:
        st.warning("Population context requires the Full HITGNN (Phase 8) model.")
        return
        
    idx = ctx.test_dataset.patient_records.index(patient)
    data = ctx.test_dataset[idx]
    
    if ctx.phase8_model is None or ctx.reference_embeddings is None:
        st.error("Model or Reference Bank unavailable.")
        return
        
    batch = next(iter(DataLoader([data], batch_size=1))).to(ctx.device)
    
    with torch.no_grad():
        out = ctx.phase8_model.encoder(batch, return_attention=False)
        query_emb = out.fused_embedding
        query_meta = batch.patient_meta
        
        sim, attn = ctx.phase8_model.population_graph(query_emb, query_meta, ctx.reference_embeddings, ctx.reference_metadata, return_attention=True)
        
        sim = sim.squeeze(0).cpu().numpy()
        attn = attn.squeeze(0).cpu().numpy()
        
    k = ctx.phase8_model.population_graph.k
    top_indices = sim.argsort()[-k:][::-1]
    
    neighbors = []
    for rank, i in enumerate(top_indices):
        n_id = ctx.reference_ids[i]
        n_idx = next(idx for idx, p in enumerate(ctx.train_dataset.patient_records) if p.patient_id == n_id)
        n_rec = ctx.train_dataset.patient_records[n_idx]
        
        age_diff = n_rec.age - patient.age
        sofa_diff = n_rec.sofa_score - patient.sofa_score
        icu_match = "Yes" if n_rec.icu_type == patient.icu_type else "No"
        
        neighbors.append({
            "Rank": rank + 1,
            "Neighbor ID": n_id,
            "Similarity": float(sim[i]),
            "Attention Weight": float(attn[i]),
            "Age Δ": f"{age_diff:+.1f}",
            "SOFA Δ": f"{sofa_diff:+.1f}",
            "ICU Match": icu_match
        })
        
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    df = pd.DataFrame(neighbors)
    st.markdown("### Retrieved TRAIN Neighbors")
    st.info("Neighbors retrieved from the frozen training set. Mortality outcome is never provided to the model or UI.")
    st.dataframe(df, use_container_width=True)
    
    st.markdown("### Population Neighborhood Graph")
    render_population_graph(patient.patient_id, neighbors)
    st.markdown('</div>', unsafe_allow_html=True)

def render_population_graph(query_id, neighbors):
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("PyVis is not installed.")
        return
        
    net = Network(height="400px", width="100%", notebook=True, cdn_resources='in_line', bgcolor="#111827", font_color="#F8FAFC")
    
    net.add_node(query_id, label=f"Query:\n{query_id}", color="#FB7185", size=25)
    
    for n in neighbors:
        n_id = n["Neighbor ID"]
        sim = n["Similarity"]
        attn = n["Attention Weight"]
        
        title = f"Similarity: {sim:.3f}\nAttention: {attn:.3f}"
        net.add_node(n_id, label=n_id, title=title, color="#6C8CFF", size=15)
        # Edge thickness ~ attention
        net.add_edge(query_id, n_id, value=attn, title=f"Sim: {sim:.2f}", color="rgba(255,255,255,0.2)")
        
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    try:
        html = net.generate_html()
        components.html(html, height=420)
    except Exception as e:
        st.error(f"Failed to render PyVis graph: {e}")
