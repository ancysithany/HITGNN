"""
src.models.hitgnn — Full HITGNN Model Assembly.

Orchestrates the complete forward pass. Phase 6 implements the PatientEncoder:

    1. Event graph  → TE-GAT layers       → node embeddings
    2. Biomarker panel → Missingness encoder → z_miss
    3. Node embeddings → Hierarchical pooling → h_p
    4. Fuse h_p with z_miss               → h_p_fused
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.models.event_node_encoder import EventNodeEncoder
from src.models.te_gat import TEGATEncoder
from src.models.hierarchical_pool import HierarchicalAttentionPool
from src.models.missingness_encoder import MissingnessEncoder, ValueOnlyEncoder


@dataclass
class PatientEncoderOutput:
    """Structured output for the patient encoding phase."""
    fused_embedding: torch.Tensor
    patient_event_embedding: torch.Tensor
    missingness_summary: torch.Tensor
    node_embeddings: torch.Tensor
    slot_missingness_embeddings: torch.Tensor
    node_attention: torch.Tensor          # Type-level attention over nodes
    type_attention: torch.Tensor          # Patient-level attention over types
    type_present_mask: torch.Tensor
    te_gat_attention: list[dict] = None   # Detailed TE-GAT edge attention records


class PatientEncoder(nn.Module):
    """Encodes a variable-size patient graph + missingness into a fixed vector.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters.
    vocab_size : int
        Number of distinct clinical concepts.
    """

    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        
        # Load hyperparameters
        gat_cfg = config.get("te_gat", {})
        pool_cfg = config.get("pooling", {})
        miss_cfg = config.get("missingness", {})
        
        node_hidden = gat_cfg.get("hidden_dim", 64)
        time_dim = gat_cfg.get("time_encoding_dim", 16)
        n_heads = gat_cfg.get("n_heads", 4)
        n_layers = gat_cfg.get("n_layers", 3)
        edge_type_dim = gat_cfg.get("edge_type_dim", 8)
        
        miss_hidden = miss_cfg.get("hidden_dim", 64)
        fused_dim = pool_cfg.get("fused_dim", 128)
        dropout = pool_cfg.get("dropout", 0.2)
        n_types = 6  # Fixed to canonical event types
        
        # Components
        self.node_encoder = EventNodeEncoder(
            num_concepts=vocab_size,
            hidden_dim=node_hidden,
            time_dim=time_dim
        )
        
        self.te_gat = TEGATEncoder(
            in_dim=node_hidden,
            hidden_dim=node_hidden,
            n_heads=n_heads,
            n_layers=n_layers,
            time_dim=time_dim,
            edge_type_dim=edge_type_dim
        )
        
        self.ablation_cfg = config.get("ablation", {})
        
        if self.ablation_cfg.get("no_missingness", False):
            self.missingness_encoder = ValueOnlyEncoder(
                hidden_dim=miss_hidden,
                output_dim=miss_hidden
            )
        else:
            self.missingness_encoder = MissingnessEncoder(
                hidden_dim=miss_hidden,
                output_dim=miss_hidden
            )
        
        self.fused_dim = fused_dim
        self.pool = HierarchicalAttentionPool(
            hidden_dim=node_hidden,
            n_types=n_types,
            missingness_dim=miss_hidden,
            fused_dim=fused_dim,
            dropout=dropout
        )

    def forward(
        self,
        batch: Batch,
        return_attention: bool = True
    ) -> PatientEncoderOutput:
        """Forward pass for patient encoding.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched PyG graphs containing event nodes, edges, and missingness tensors.
        return_attention : bool
            If True, retrieves internal attention weights.

        Returns
        -------
        PatientEncoderOutput
        """
        # 1. Missingness Branch
        slot_embeddings, z_miss = self.missingness_encoder(batch.missingness_input)
        
        # 2. Graph Branch - Node Encoding
        h0 = self.node_encoder(batch)
        
        # 3. TE-GAT Contextualization
        h_nodes, te_gat_attn = self.te_gat(
            h0,
            batch.edge_index,
            batch.edge_delta_t,
            batch.edge_type,
            return_attention=return_attention,
            static_gnn=self.ablation_cfg.get("static_gnn", False)
        )
        
        # 4. Hierarchical Pooling and Fusion
        if self.ablation_cfg.get("mean_pooling", False):
            from torch_geometric.nn import global_mean_pool
            h_p = global_mean_pool(h_nodes, batch.batch)
            
            # Simple fusion
            if not hasattr(self, "mean_pool_fusion"):
                self.mean_pool_fusion = nn.Linear(h_nodes.size(-1) + z_miss.size(-1), self.fused_dim).to(h_p.device)
            fused = self.mean_pool_fusion(torch.cat([h_p, z_miss], dim=-1))
            
            # Dummies for return
            type_embeddings = None
            type_present_mask = None
            beta = None
            alpha = None
        else:
            (
                fused,
                h_p,
                type_embeddings,
                type_present_mask,
                beta,
                alpha
            ) = self.pool(
                h=h_nodes,
                node_type=batch.node_type,
                batch_index=batch.batch,
                batch_size=batch.num_graphs,
                z_miss=z_miss
            )
        
        return PatientEncoderOutput(
            fused_embedding=fused,
            patient_event_embedding=h_p,
            missingness_summary=z_miss,
            node_embeddings=h_nodes,
            slot_missingness_embeddings=slot_embeddings,
            node_attention=alpha,
            type_attention=beta,
            type_present_mask=type_present_mask,
            te_gat_attention=te_gat_attn
        )


from src.models.prediction_head import PredictionHead

class PatientMortalityModel(nn.Module):
    """Patient-level Phase 7 mortality model.
    
    Architecture: PatientEncoder -> PredictionHead -> logit
    """
    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        self.encoder = PatientEncoder(config, vocab_size)
        
        pool_cfg = config.get("pooling", {})
        fused_dim = pool_cfg.get("fused_dim", 128)
        
        pred_cfg = config.get("prediction_head", {})
        hidden_dim = pred_cfg.get("hidden_dim", 64)
        dropout = pred_cfg.get("dropout", 0.3)
        
        self.prediction_head = PredictionHead(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, batch: Batch, return_attention: bool = False) -> torch.Tensor:
        """Forward pass for patient-level mortality prediction.
        
        Returns:
            torch.Tensor: Raw mortality logits, shape [N] (if return_attention=False)
            tuple: (logits, attention_dict) (if return_attention=True)
        """
        out = self.encoder(batch, return_attention=return_attention)
        logit = self.prediction_head(out.fused_embedding)
        if return_attention:
            attn_dict = {
                "node_attention": out.node_attention,
                "type_attention": out.type_attention,
                "type_present_mask": out.type_present_mask,
                "te_gat_attention": out.te_gat_attention
            }
            return logit, attn_dict
        return logit


class HITGNN(nn.Module):
    """Hierarchical Temporal Event Graph Neural Network.

    Placeholder for future phases (population graph & prediction).
    Currently just wraps PatientMortalityModel.
    """
    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        self.model = PatientMortalityModel(config, vocab_size)

    def forward(self, batch: Batch, return_attention: bool = False):
        return self.model(batch, return_attention=return_attention)

from src.models.population_graph import PopulationGraphModule

class PopulationHITGNN(nn.Module):
    """Phase 8 Population-Level Mortality Model.
    
    Architecture: PatientEncoder -> PopulationGraphModule -> PredictionHead -> logit
    """
    def __init__(self, config: dict, vocab_size: int) -> None:
        super().__init__()
        self.encoder = PatientEncoder(config, vocab_size)
        
        self.population_module = PopulationGraphModule(config)
        
        pool_cfg = config.get("pooling", {})
        fused_dim = pool_cfg.get("fused_dim", 128)
        
        pred_cfg = config.get("prediction_head", {})
        hidden_dim = pred_cfg.get("hidden_dim", 64)
        dropout = pred_cfg.get("dropout", 0.3)
        
        self.prediction_head = PredictionHead(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(
        self, 
        batch: Batch, 
        reference_embeddings: torch.Tensor,
        reference_metadata: torch.Tensor,
        reference_ids: list[str],
        return_attention: bool = False
    ):
        """Forward pass for population-level mortality prediction.
        
        Returns:
            torch.Tensor: Raw mortality logits, shape [N] (if return_attention=False)
            tuple: (logits, attention_dict) (if return_attention=True)
        """
        # 1. Patient Encoding
        out = self.encoder(batch, return_attention=return_attention)
        h_p = out.fused_embedding
        
        # 2. Population Reasoning
        h_prime, topk_idx, topk_sim, pop_alpha = self.population_module(
            query_embeddings=h_p,
            query_metadata=batch.patient_meta,
            reference_embeddings=reference_embeddings,
            reference_metadata=reference_metadata,
            query_ids=batch.patient_id,
            reference_ids=reference_ids
        )
        
        # 3. Prediction
        logit = self.prediction_head(h_prime)
        if return_attention:
            attn_dict = {
                "node_attention": out.node_attention,
                "type_attention": out.type_attention,
                "type_present_mask": out.type_present_mask,
                "te_gat_attention": out.te_gat_attention,
                "population_attention": pop_alpha,
                "population_topk_idx": topk_idx,
                "population_topk_sim": topk_sim
            }
            return logit, attn_dict
        return logit
