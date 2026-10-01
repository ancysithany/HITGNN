"""
scripts/visualize_population.py — Phase 8 Population Graph Visualization.
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def main():
    csv_path = "outputs/metrics/population_neighbors.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSV is empty.")
        return
        
    # Pick the first TEST query
    q_id = df.iloc[0]["query_patient_id"]
    sub_df = df[df["query_patient_id"] == q_id]
    
    G = nx.DiGraph()
    G.add_node(q_id, type="query")
    
    for _, row in sub_df.iterrows():
        r_id = row["neighbor_patient_id"]
        sim = row["similarity_score"]
        G.add_node(r_id, type="reference")
        G.add_edge(q_id, r_id, weight=sim)
        
    plt.figure(figsize=(8, 6))
    
    pos = nx.spring_layout(G, seed=42)
    
    query_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "query"]
    ref_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "reference"]
    
    nx.draw_networkx_nodes(G, pos, nodelist=query_nodes, node_color='red', node_size=800, label="Query (TEST)")
    nx.draw_networkx_nodes(G, pos, nodelist=ref_nodes, node_color='blue', node_size=500, label="Neighbors (TRAIN)")
    
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=9)
    
    weights = [d["weight"] for u, v, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6, edge_color=weights, edge_cmap=plt.cm.Blues, edge_vmin=min(weights)-0.1, edge_vmax=max(weights))
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Population Graph for Query {q_id}")
    plt.legend(scatterpoints=1)
    plt.axis("off")
    
    os.makedirs("outputs/figures", exist_ok=True)
    out_path = "outputs/figures/example_population_graph.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
