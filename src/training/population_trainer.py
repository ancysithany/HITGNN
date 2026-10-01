"""
src.training.population_trainer — Trainer for Phase 8 Population Graph Model.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np

from src.training.metrics import calculate_metrics

class PopulationTrainer:
    """Trainer for Phase 8 Population-Level Mortality Model.
    
    Implements inductive training strategy:
    - Train queries retrieve from Train reference bank
    - Val queries retrieve from frozen Train reference bank
    - Test queries retrieve from frozen Train reference bank
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        meta_normalizer: 'PatientMetaNormalizer'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.meta_normalizer = meta_normalizer
        
        train_cfg = config.get("training", {})
        
        self.loss_type = train_cfg.get("loss", "bce_with_logits")
        pos_weight_cfg = train_cfg.get("pos_weight", "auto")
        
        self.pos_weight = None
        if pos_weight_cfg == "auto":
            num_pos = 0
            num_total = 0
            for batch in self.train_loader:
                y = batch.y
                num_pos += y.sum().item()
                num_total += y.size(0)
            num_neg = num_total - num_pos
            if num_pos > 0:
                self.pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device)
            else:
                self.pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
        elif isinstance(pos_weight_cfg, (int, float)):
            self.pos_weight = torch.tensor([pos_weight_cfg], dtype=torch.float32, device=device)
            
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        lr = train_cfg.get("learning_rate", 0.001)
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max' if train_cfg.get("monitor_metric", "auprc") in ["auroc", "auprc"] else 'min',
            factor=train_cfg.get("scheduler_factor", 0.5),
            patience=train_cfg.get("scheduler_patience", 7),
            verbose=True
        )
        
        self.epochs = train_cfg.get("epochs", 100)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 15)
        self.monitor_metric = train_cfg.get("monitor_metric", "auprc")
        self.gradient_clip_norm = train_cfg.get("gradient_clip_norm", 5.0)
        
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "outputs/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_metric = -float('inf') if self.scheduler.mode == 'max' else float('inf')
        self.best_epoch = -1
        self.patience_counter = 0

    @torch.no_grad()
    def build_reference_bank(self):
        """Computes current embeddings and metadata for ALL training patients."""
        self.model.eval()
        all_emb = []
        all_meta = []
        all_ids = []
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            out = self.model.encoder(batch, return_attention=False)
            h_p = out.fused_embedding
            
            all_emb.append(h_p.cpu())
            all_meta.append(batch.patient_meta.cpu())
            all_ids.extend(batch.patient_id)
            
        return (
            torch.cat(all_emb, dim=0).to(self.device),
            torch.cat(all_meta, dim=0).to(self.device),
            all_ids
        )

    def train_epoch(self):
        # 1. Build detached reference bank for the epoch
        ref_emb, ref_meta, ref_ids = self.build_reference_bank()
        
        self.model.train()
        total_loss = 0.0
        
        y_true = []
        y_prob = []
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            logits = self.model(
                batch=batch,
                reference_embeddings=ref_emb,
                reference_metadata=ref_meta,
                reference_ids=ref_ids
            )
            
            y = batch.y.float()
            if logits.ndim == 1:
                logits = logits.view(-1)
                y = y.view(-1)
                
            loss = self.criterion(logits, y)
            loss.backward()
            
            if self.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                y_true.extend(y.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                
        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = calculate_metrics(np.array(y_true), np.array(y_prob))
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, ref_emb=None, ref_meta=None, ref_ids=None, threshold=0.5):
        if ref_emb is None:
            ref_emb, ref_meta, ref_ids = self.build_reference_bank()
            
        self.model.eval()
        total_loss = 0.0
        
        y_true = []
        y_prob = []
        all_logits = []
        
        all_neighbor_indices = []
        all_neighbor_similarities = []
        all_neighbor_attn = []
        all_query_ids = []
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Forward pass specifically capturing outputs from the population module
            out = self.model.encoder(batch, return_attention=False)
            h_p = out.fused_embedding
            
            h_prime, topk_idx, topk_sim, alpha = self.model.population_module(
                query_embeddings=h_p,
                query_metadata=batch.patient_meta,
                reference_embeddings=ref_emb,
                reference_metadata=ref_meta,
                query_ids=batch.patient_id,
                reference_ids=ref_ids
            )
            
            logits = self.model.prediction_head(h_prime)
            
            y = batch.y.float()
            if logits.ndim == 1:
                logits = logits.view(-1)
                y = y.view(-1)
                
            loss = self.criterion(logits, y)
            total_loss += loss.item() * batch.num_graphs
            
            probs = torch.sigmoid(logits)
            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            all_neighbor_indices.append(topk_idx.cpu())
            all_neighbor_similarities.append(topk_sim.cpu())
            all_neighbor_attn.append(alpha.cpu())
            all_query_ids.extend(batch.patient_id)
            
        avg_loss = total_loss / len(loader.dataset)
        
        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)
        all_logits_np = np.array(all_logits)
        
        metrics = calculate_metrics(y_true_np, y_prob_np, threshold)
        
        # Merge lists of tensors into big tensors if we actually need them
        if all_neighbor_indices:
            all_neighbor_indices = torch.cat(all_neighbor_indices, dim=0)
            all_neighbor_similarities = torch.cat(all_neighbor_similarities, dim=0)
            all_neighbor_attn = torch.cat(all_neighbor_attn, dim=0)
            
        return (
            avg_loss, metrics, y_true_np, y_prob_np, all_logits_np,
            all_query_ids, all_neighbor_indices, all_neighbor_similarities, all_neighbor_attn
        )

    def train(self):
        print(f"Starting population training for {self.epochs} epochs")
        
        history = {'train_loss': [], 'val_loss': [], 'train_auroc': [], 'val_auroc': []}
            
        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics, *_ = self.evaluate(self.val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auroc'].append(train_metrics['auroc'])
            history['val_auroc'].append(val_metrics['auroc'])
            
            monitor_val = val_metrics[self.monitor_metric] if self.monitor_metric in val_metrics else val_loss
            self.scheduler.step(monitor_val)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:02d}/{self.epochs} | Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | Val AUPRC: {val_metrics['auprc']:.4f} | LR: {current_lr:.6f}")
            
            is_best = False
            if self.scheduler.mode == 'max':
                if monitor_val > self.best_metric:
                    self.best_metric = monitor_val
                    is_best = True
            else:
                if monitor_val < self.best_metric:
                    self.best_metric = monitor_val
                    is_best = True
                    
            if is_best:
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint()
                print("  [*] New best model saved.")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        print(f"Training complete. Best epoch: {self.best_epoch} with {self.monitor_metric}={self.best_metric:.4f}")
        
    def save_checkpoint(self, filename="best_population_model.pt"):
        path = os.path.join(self.checkpoint_dir, filename)
        
        state = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_validation_metric': self.best_metric,
            'config': self.config,
            'pos_weight': self.pos_weight.item() if self.pos_weight is not None else None,
            'meta_normalizer_state': self.meta_normalizer.state_dict()
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_validation_metric', -float('inf'))
        self.meta_normalizer.load_state_dict(checkpoint.get('meta_normalizer_state', {}))
        print(f"Resumed from checkpoint: {path} (Epoch {self.best_epoch})")
