"""
src.visualization.graph — Visualize patient graphs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.graph.graph_builder import (
    EDGE_TYPE_TEMPORAL_ADJACENCY, EDGE_TYPE_SAME_CONCEPT,
    EDGE_TYPE_SEMANTIC, EDGE_TYPE_CROSS_MODALITY,
    NODE_TYPE_MAP
)

def visualize_patient_graph(data: Data, save_path: str):
    """Visualize a Phase 4 patient graph using NetworkX."""
    # Convert to networkx
    G = to_networkx(data, to_undirected=False, node_attrs=["node_type", "concept_id", "time"])
    
    # Setup plot
    plt.figure(figsize=(12, 8))
    
    # We can layout using time as X-axis and node_type as Y-axis
    pos = {}
    for n, d in G.nodes(data=True):
        x = d["time"]
        y = d["node_type"]
        # Add slight jitter to avoid exact overlap
        pos[n] = (x, y + (n % 5) * 0.1)

    # Node colors by type
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    node_color = [colors[d["node_type"]] for _, d in G.nodes(data=True)]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=100, alpha=0.8)

    # Draw edges by type
    u_nodes = data.edge_index[0].numpy()
    v_nodes = data.edge_index[1].numpy()
    edge_types = data.edge_type.numpy()
    
    e_temp = [(u, v) for u, v, et in zip(u_nodes, v_nodes, edge_types) if et == EDGE_TYPE_TEMPORAL_ADJACENCY]
    e_rec = [(u, v) for u, v, et in zip(u_nodes, v_nodes, edge_types) if et == EDGE_TYPE_SAME_CONCEPT]
    e_sem = [(u, v) for u, v, et in zip(u_nodes, v_nodes, edge_types) if et == EDGE_TYPE_SEMANTIC]
    e_cross = [(u, v) for u, v, et in zip(u_nodes, v_nodes, edge_types) if et == EDGE_TYPE_CROSS_MODALITY]

    nx.draw_networkx_edges(G, pos, edgelist=e_temp, edge_color="black", arrows=True, alpha=0.5, style="solid")
    nx.draw_networkx_edges(G, pos, edgelist=e_rec, edge_color="blue", arrows=True, alpha=0.5, style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=e_sem, edge_color="green", arrows=True, alpha=0.5, style="dotted")
    nx.draw_networkx_edges(G, pos, edgelist=e_cross, edge_color="orange", arrows=True, alpha=0.5, style="dashdot")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Temporal Adjacency", linestyle="solid"),
        Line2D([0], [0], color="blue", lw=2, label="Recurrence", linestyle="dashed"),
        Line2D([0], [0], color="green", lw=2, label="Semantic", linestyle="dotted"),
        Line2D([0], [0], color="orange", lw=2, label="Cross Modality", linestyle="dashdot"),
    ]
    
    # Node legend
    rev_map = {v: k for k, v in NODE_TYPE_MAP.items()}
    for k, v in rev_map.items():
        legend_elements.append(Line2D([0], [0], marker="o", color="w", label=v, markerfacecolor=colors[k], markersize=10))

    plt.legend(handles=legend_elements, loc="best", fontsize=8)
    plt.title(f"Patient Event Graph (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
    plt.xlabel("Time relative to t0 (hours)")
    plt.ylabel("Node Category")
    plt.grid(True, linestyle="--", alpha=0.3)
    
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
