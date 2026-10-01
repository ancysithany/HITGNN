import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.utils.context import DashboardContext
from src.data.canonical import PatientRecord

def render_timeline(ctx: DashboardContext, patient: PatientRecord):
    st.title("Clinical Timeline")
    
    st.markdown('<div class="hitgnn-card">', unsafe_allow_html=True)
    events = patient.events
    
    if not events:
        st.info("No events found for this patient.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    rows = []
    for ev in events:
        rows.append({
            "concept": ev.concept_name,
            "event_type": ev.event_type,
            "relative_time_hours": ev.timestamp,
            "value": ev.value if ev.value is not None else "N/A",
            "unit": ev.unit if ev.unit is not None else ""
        })
        
    df = pd.DataFrame(rows)
    
    # Sort for consistent rendering
    df = df.sort_values(by=["event_type", "relative_time_hours"])
    
    fig = px.scatter(
        df, 
        x="relative_time_hours", 
        y="event_type", 
        color="event_type",
        hover_data=["concept", "value", "unit"],
        labels={"relative_time_hours": "Hours relative to t0", "event_type": "Event Type"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Add vertical t0 marker
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#FACC15", annotation_text="t0 = Sepsis recognition", annotation_position="top right")
    
    fig.update_layout(
        height=500,
        plot_bgcolor="#111827",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
