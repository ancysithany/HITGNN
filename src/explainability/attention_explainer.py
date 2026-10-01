"""
src.explainability.attention_explainer - Extract and interpret hierarchical attention.
"""

import torch
from typing import Dict, Any, List

def extract_hierarchical_attention(
    patient,
    batch,
    model,
    reference_embeddings=None,
    reference_metadata=None,
    reference_ids=None
) -> Dict[str, Any]:
    """
    Extracts node-level and type-level attention from the model.
    """
    model.eval()
    
    # Check if PopulationHITGNN or PatientMortalityModel
    if reference_embeddings is not None:
        with torch.no_grad():
            logit, attn_dict = model(
                batch, 
                reference_embeddings=reference_embeddings,
                reference_metadata=reference_metadata,
                reference_ids=reference_ids,
                return_attention=True
            )
    else:
        with torch.no_grad():
            logit, attn_dict = model(batch, return_attention=True)
            
    # Re-build sorted events to map nodes back
    events = [e for e in patient.events if e.timestamp <= 0.0]
    
    # Same sorting logic as graph_builder.py
    NODE_TYPE_MAP = {
        "LAB": 0, "VITAL": 1, "MEDICATION": 2, 
        "PROCEDURE": 3, "DIAGNOSIS": 4, "CLINICAL_CONCEPT": 5
    }
    annotated_events = [
        (e.timestamp, NODE_TYPE_MAP[e.event_type], e.concept_id, i, e)
        for i, e in enumerate(events)
    ]
    annotated_events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    sorted_events = [x[4] for x in annotated_events]
    
    node_attention = attn_dict["node_attention"]
    type_attention = attn_dict["type_attention"]
    if type_attention is not None:
        type_attention = type_attention[0]
        
    event_importance = []
    
    if type_attention is not None and node_attention is not None:
        # Map node attention
        for i, e in enumerate(sorted_events):
            t_idx = NODE_TYPE_MAP[e.event_type]
            t_attn = type_attention[t_idx].item()
            n_attn = node_attention[i].item()
            event_importance.append({
                "patient_id": patient.patient_id,
                "event_index": i,
                "event_type": e.event_type,
                "concept": e.concept_name,
                "relative_time": e.timestamp,
                "value": e.value,
                "within_type_attention": n_attn,
                "patient_type_attention": t_attn,
                "hierarchical_attention_score": n_attn * t_attn
            })
            
    type_importance = {}
    if type_attention is not None:
        for t_name, t_idx in NODE_TYPE_MAP.items():
            type_importance[t_name] = type_attention[t_idx].item()
            
    population_neighbors = []
    if "population_attention" in attn_dict and attn_dict["population_attention"] is not None:
        pop_alpha = attn_dict["population_attention"][0]
        topk_idx = attn_dict["population_topk_idx"][0]
        topk_sim = attn_dict["population_topk_sim"][0]
        for rank, (idx, sim, alpha) in enumerate(zip(topk_idx, topk_sim, pop_alpha)):
            if sim > -1e5:  # skip invalid
                neighbor_id = reference_ids[idx.item()]
                population_neighbors.append({
                    "neighbor_rank": rank + 1,
                    "neighbor_patient_id": neighbor_id,
                    "similarity": sim.item(),
                    "population_attention_weight": alpha.item()
                })
                
    return {
        "logit": logit.item(),
        "event_importance": event_importance,
        "type_importance": type_importance,
        "population_neighbors": population_neighbors
    }
