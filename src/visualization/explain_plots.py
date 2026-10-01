"""
src.visualization.explain_plots - Plots for explainability outputs.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

def plot_type_importance(type_importance: dict, filepath: str):
    plt.figure(figsize=(8, 5))
    types = list(type_importance.keys())
    vals = list(type_importance.values())
    
    plt.bar(types, vals, color='skyblue')
    plt.title("Event Type Attention Importance")
    plt.ylabel("Attention Weight")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_timeline(event_importance: list, filepath: str):
    if not event_importance:
        # Save empty plot
        plt.figure(figsize=(10, 4))
        plt.title("No Events to Plot")
        plt.savefig(filepath)
        plt.close()
        return
        
    df = pd.DataFrame(event_importance)
    
    plt.figure(figsize=(10, 6))
    
    # Plot stems
    plt.stem(df['relative_time'], df['hierarchical_attention_score'], basefmt=" ")
    
    # Annotate top 5 events
    top5 = df.nlargest(5, 'hierarchical_attention_score')
    for _, row in top5.iterrows():
        plt.annotate(
            row['concept'],
            (row['relative_time'], row['hierarchical_attention_score']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
        
    plt.title(f"Temporal Event Importance (Patient {df['patient_id'].iloc[0]})")
    plt.xlabel("Time relative to prediction (hours)")
    plt.ylabel("Hierarchical Attention Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_missingness_importance(missingness_attribution: torch.Tensor, filepath: str):
    pass # Wait, not requested as standalone plot. We need `plot_shap_missingness` instead.

def plot_population_attention(population_neighbors: list, filepath: str):
    if not population_neighbors:
        # Save empty plot
        plt.figure(figsize=(8, 4))
        plt.title("No Population Graph Used")
        plt.savefig(filepath)
        plt.close()
        return
        
    df = pd.DataFrame(population_neighbors)
    df = df.sort_values("population_attention_weight", ascending=False)
    
    plt.figure(figsize=(8, 5))
    plt.bar(df['neighbor_patient_id'], df['population_attention_weight'], color='coral')
    plt.title("Population Neighbor Attention Weights")
    plt.ylabel("Attention Weight")
    plt.xlabel("Neighbor Patient ID")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_shap_metadata(shap_values: dict, filepath: str):
    df = pd.DataFrame.from_dict(shap_values, orient='index')
    if df.empty:
        return
    df = df.sort_values("abs_shap_value", ascending=True)
    
    plt.figure(figsize=(8, 6))
    colors = ['green' if x > 0 else 'red' for x in df['shap_value']]
    plt.barh(df.index, df['shap_value'], color=colors)
    plt.title("SHAP Attribution (Metadata)")
    plt.xlabel("SHAP Value")
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_shap_missingness(shap_values: dict, filepath: str):
    df = pd.DataFrame.from_dict(shap_values, orient='index')
    if df.empty:
        return
    
    # Take top 20 most important features
    df = df.sort_values("abs_shap_value", ascending=True).tail(20)
    
    plt.figure(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in df['shap_value']]
    plt.barh(df.index, df['shap_value'], color=colors)
    plt.title("Top 20 SHAP Attribution (Missingness)")
    plt.xlabel("SHAP Value")
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
