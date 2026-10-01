"""
src.models.population_graph — Inductive Population Graph Reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PopulationGraphModule(nn.Module):
    """Inductive Population Graph Module for HITGNN.
    
    Contextualizes query patient embeddings using a reference bank of training patients.
    """
    
    def __init__(self, config: dict) -> None:
        super().__init__()
        
        pop_cfg = config.get("population_graph", {})
        pool_cfg = config.get("pooling", {})
        
        self.k_neighbors = pop_cfg.get("k_neighbors", 10)
        self.lambda_embed = pop_cfg.get("lambda_embed", 0.7)
        self.lambda_meta = pop_cfg.get("lambda_meta", 0.3)
        self.hidden_dim = pool_cfg.get("fused_dim", 128)
        
        # Attention scoring MLP: [query || ref || similarity] -> score
        # Dim: hidden_dim + hidden_dim + 1
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.W_neighbor = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def _compute_similarity(
        self, 
        q_emb: torch.Tensor, 
        q_meta: torch.Tensor, 
        ref_emb: torch.Tensor, 
        ref_meta: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity matrix between queries and references.
        
        Args:
            q_emb: [Q, D]
            q_meta: [Q, 10]
            ref_emb: [R, D]
            ref_meta: [R, 10]
            
        Returns:
            sim: [Q, R] similarity scores
        """
        Q = q_emb.size(0)
        R = ref_emb.size(0)
        
        # 1. Cosine similarity
        # Expand for broadcasting: q_emb [Q, 1, D], ref_emb [1, R, D]
        q_emb_exp = q_emb.unsqueeze(1)
        ref_emb_exp = ref_emb.unsqueeze(0)
        cos_sim = F.cosine_similarity(q_emb_exp, ref_emb_exp, dim=2, eps=1e-8) # [Q, R]
        
        # 2. Metadata similarity: exp(-||delta_p - delta_q||_2^2)
        q_meta_exp = q_meta.unsqueeze(1)
        ref_meta_exp = ref_meta.unsqueeze(0)
        
        sq_dist = torch.sum((q_meta_exp - ref_meta_exp) ** 2, dim=2) # [Q, R]
        meta_sim = torch.exp(-sq_dist) # [Q, R]
        
        sim = self.lambda_embed * cos_sim + self.lambda_meta * meta_sim
        return sim
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        query_metadata: torch.Tensor,
        reference_embeddings: torch.Tensor,
        reference_metadata: torch.Tensor,
        query_ids: Optional[list[str]] = None,
        reference_ids: Optional[list[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for population graph reasoning.
        
        Args:
            query_embeddings: [Q, D]
            query_metadata: [Q, 10]
            reference_embeddings: [R, D]
            reference_metadata: [R, 10]
            query_ids: list of string IDs for queries
            reference_ids: list of string IDs for references
            
        Returns:
            contextualized_embeddings: [Q, D]
            neighbor_indices: [Q, k]
            neighbor_similarities: [Q, k]
            attention_weights: [Q, k]
        """
        Q = query_embeddings.size(0)
        R = reference_embeddings.size(0)
        
        if R == 0:
            # Degenerate case: no references
            device = query_embeddings.device
            return query_embeddings, torch.empty((Q,0), device=device), torch.empty((Q,0), device=device), torch.empty((Q,0), device=device)
            
        # 1. Compute full similarity matrix
        sim_matrix = self._compute_similarity(
            query_embeddings, query_metadata, reference_embeddings, reference_metadata
        ) # [Q, R]
        
        # 2. Self-exclusion for training queries
        if query_ids is not None and reference_ids is not None:
            for i, q_id in enumerate(query_ids):
                for j, r_id in enumerate(reference_ids):
                    if q_id == r_id:
                        sim_matrix[i, j] = -float('inf')
                        
        # 3. Top-k Retrieval
        effective_k = min(self.k_neighbors, R)
        if query_ids is not None and reference_ids is not None:
            # If a patient is its own only neighbor, effective_k might need adjustment 
            # if we have exactly self.k_neighbors references and self is one.
            # But topk handles it by picking the next best. If R <= self.k_neighbors and we mask one, 
            # we might still get -inf if we ask for effective_k elements.
            # Let's count valid neighbors.
            pass
            
        # To be completely safe:
        # Sort and take top k
        topk_sim, topk_idx = torch.topk(sim_matrix, k=effective_k, dim=1) # [Q, k]
        
        # Mask out any -inf that made it into topk (if R <= effective_k and we masked out self)
        valid_mask = topk_sim > -1e9
        
        # 4. Message Passing (Attention)
        # Gather neighbor embeddings
        # ref_emb: [R, D] -> [Q, k, D]
        # We can do this efficiently via indexing
        # Expand queries to match [Q, k, D]
        q_exp = query_embeddings.unsqueeze(1).expand(-1, effective_k, -1) # [Q, k, D]
        
        # Gather neighbors
        flat_idx = topk_idx.flatten() # [Q*k]
        gathered_refs = reference_embeddings[flat_idx].view(Q, effective_k, -1) # [Q, k, D]
        gathered_sims = topk_sim.unsqueeze(2) # [Q, k, 1]
        
        # Attention scoring
        attn_input = torch.cat([q_exp, gathered_refs, gathered_sims], dim=2) # [Q, k, 2D+1]
        e = self.attn_mlp(attn_input).squeeze(2) # [Q, k]
        
        # Mask out invalid neighbors (like self)
        e = e.masked_fill(~valid_mask, -1e9)
        
        # Softmax
        alpha = F.softmax(e, dim=1) # [Q, k]
        # To prevent NaN if all are invalid
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
        
        # Context aggregation
        # alpha: [Q, k], gathered_refs: [Q, k, D]
        W_refs = self.W_neighbor(gathered_refs) # [Q, k, D]
        alpha_exp = alpha.unsqueeze(2) # [Q, k, 1]
        
        c = torch.sum(alpha_exp * W_refs, dim=1) # [Q, D]
        
        # 5. Residual Update
        h_prime = self.layer_norm(query_embeddings + self.W_context(c))
        
        return h_prime, topk_idx, topk_sim, alpha
