import streamlit as st
import streamlit.components.v1 as components
from dashboard.utils.context import DashboardContext
from src.data.canonical import PatientRecord

def render_graph(ctx: DashboardContext, patient: PatientRecord, selected_model: str):
    st.title("Event Graph")
    
    idx = ctx.test_dataset.patient_records.index(patient)
    model_graph = ctx.test_dataset[idx]
    
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Controls")
        
        edge_types_selected = st.multiselect(
            "Edge Types",
            options=["TEMPORAL_ADJACENCY", "SAME_CONCEPT_RECURRENCE", "SEMANTIC", "CROSS_MODALITY"],
            default=["TEMPORAL_ADJACENCY", "SAME_CONCEPT_RECURRENCE", "SEMANTIC", "CROSS_MODALITY"]
        )
        
        max_nodes = st.slider("Max Nodes to Render", 10, 500, 100)
        time_window = st.slider("Time Window (hours before t0)", 0, 240, 240)
        
        min_attention = st.slider("Minimum attention for display", 0.0, 1.0, 0.0, 0.05)
        
        st.info("Visualization filtering does not affect model inference. Model inference uses the complete graph.")
        
        st.markdown("#### Modality Legend")
        st.markdown("""
        <ul style='list-style-type: none; padding-left: 0; color: #94A3B8; font-size: 0.9em;'>
            <li><span style='color: #FB7185;'>■</span> LAB (Square)</li>
            <li><span style='color: #6C8CFF;'>▲</span> VITAL (Triangle)</li>
            <li><span style='color: #4ADE80;'>●</span> MEDICATION (Dot)</li>
            <li><span style='color: #C084FC;'>★</span> PROCEDURE (Star)</li>
            <li><span style='color: #FACC15;'>◆</span> DIAGNOSIS (Diamond)</li>
            <li><span style='color: #94A3B8;'>▼</span> CLINICAL_CONCEPT (Triangle Down)</li>
        </ul>
        """, unsafe_allow_html=True)
        
    with col2:
        render_pyvis_graph(model_graph, edge_types_selected, max_nodes, time_window, min_attention, patient)
        
    st.markdown("</div>", unsafe_allow_html=True)

def render_pyvis_graph(model_graph, edge_types_selected, max_nodes, time_window, min_attention, patient):
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("PyVis is not installed. Graph rendering unavailable.")
        return
        
    edge_index = model_graph.edge_index.numpy()
    edge_type = model_graph.edge_type.numpy()
    
    # We do not have attention available here easily unless we ran inference and extracted it.
    # We'll just filter visually. The instruction is to NEVER modify the model graph.
    
    type_map = {
        0: "TEMPORAL_ADJACENCY",
        1: "SAME_CONCEPT_RECURRENCE",
        2: "SEMANTIC",
        3: "CROSS_MODALITY"
    }
    
    valid_nodes = []
    for i, ev in enumerate(patient.events):
        # timestamp is relative to t0 (0 is t0, negative is before t0).
        # We assume time_window means anything within X hours of t0 (so >= -time_window)
        if ev.timestamp >= -time_window:
            valid_nodes.append(i)
            
    if len(valid_nodes) > max_nodes:
        valid_nodes = valid_nodes[-max_nodes:]
        
    valid_nodes_set = set(valid_nodes)
    
    net = Network(height="600px", width="100%", notebook=True, cdn_resources='in_line', bgcolor="#111827", font_color="#F8FAFC")
    
    shape_map = {
        "LAB": "square",
        "VITAL": "triangle",
        "MEDICATION": "dot",
        "PROCEDURE": "star",
        "DIAGNOSIS": "diamond",
        "CLINICAL_CONCEPT": "triangleDown"
    }
    
    color_map = {
        "LAB": "#FB7185",
        "VITAL": "#6C8CFF",
        "MEDICATION": "#4ADE80",
        "PROCEDURE": "#C084FC",
        "DIAGNOSIS": "#FACC15",
        "CLINICAL_CONCEPT": "#94A3B8"
    }
    
    for i in valid_nodes:
        ev = patient.events[i]
        title = f"Type: {ev.event_type}\nConcept: {ev.concept_name}\nTime: {ev.timestamp}h\nValue: {ev.value} {ev.unit}"
        shape = shape_map.get(ev.event_type, "dot")
        color = color_map.get(ev.event_type, "#94A3B8")
        net.add_node(i, label=ev.concept_name[:10], title=title, color=color, shape=shape)
        
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i]
        v = edge_index[1, i]
        t = edge_type[i]
        
        t_name = type_map.get(t, "UNKNOWN")
        if t_name in edge_types_selected and u in valid_nodes_set and v in valid_nodes_set:
            edge_color = "rgba(255,255,255,0.2)"
            if t == 1: edge_color = "rgba(108,140,255,0.5)"
            elif t == 2: edge_color = "rgba(74,222,128,0.5)"
            elif t == 3: edge_color = "rgba(250,204,21,0.5)"
            net.add_edge(int(u), int(v), title=t_name, color=edge_color)
            
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -10000
        }
      }
    }
    """)
    try:
        html = net.generate_html()
        components.html(html, height=620)
    except Exception as e:
        st.error(f"Failed to render PyVis graph: {e}")
