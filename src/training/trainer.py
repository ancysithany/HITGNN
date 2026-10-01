"""
src.training.trainer — Phase 7 Model Training Loop.
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np

from src.training.metrics import calculate_metrics, find_best_threshold


class Trainer:
    """Trainer for Phase 7 Patient-Level Mortality Model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        train_cfg = config.get("training", {})
        
        # Loss setup
        self.loss_type = train_cfg.get("loss", "bce_with_logits")
        pos_weight_cfg = train_cfg.get("pos_weight", "auto")
        
        self.pos_weight = None
        if pos_weight_cfg == "auto":
            # Compute from train_loader
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
            
        if self.loss_type == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not supported.")
            
        # Optimizer setup
        opt_type = train_cfg.get("optimizer", "adamw").lower()
        lr = train_cfg.get("learning_rate", 0.001)
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        
        if opt_type == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {opt_type}")
            
        # Scheduler setup
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

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        y_true = []
        y_prob = []
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            logits = self.model(batch)
            y = batch.y.float()
            
            # Ensure logits shape matches y shape
            if logits.ndim == 1:
                logits = logits.view(-1)
                y = y.view(-1)
            
            loss = self.criterion(logits, y)
            
            if not torch.isfinite(loss):
                raise RuntimeError("Loss is non-finite during training.")
                
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
    def evaluate(self, loader: DataLoader, threshold: float = 0.5):
        self.model.eval()
        total_loss = 0.0
        
        y_true = []
        y_prob = []
        all_logits = []
        
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
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
            
        avg_loss = total_loss / len(loader.dataset)
        
        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)
        all_logits_np = np.array(all_logits)
        
        metrics = calculate_metrics(y_true_np, y_prob_np, threshold)
        
        return avg_loss, metrics, y_true_np, y_prob_np, all_logits_np

    def train(self):
        print(f"Starting training for {self.epochs} epochs")
        if self.pos_weight is not None:
            print(f"Using pos_weight: {self.pos_weight.item():.4f}")
            
        history = {'train_loss': [], 'val_loss': [], 'train_auroc': [], 'val_auroc': []}
            
        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics, _, _, _ = self.evaluate(self.val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auroc'].append(train_metrics['auroc'])
            history['val_auroc'].append(val_metrics['auroc'])
            
            # Scheduling
            monitor_val = val_metrics[self.monitor_metric] if self.monitor_metric in val_metrics else val_loss
            self.scheduler.step(monitor_val)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:02d}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train AUROC: {train_metrics['auroc']: .4f} | Val AUROC: {val_metrics['auroc']:.4f}")
            print(f"  Train AUPRC: {train_metrics['auprc']: .4f} | Val AUPRC: {val_metrics['auprc']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Early stopping & checkpointing
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
        
        # Save training curves
        self._plot_history(history)

    def _plot_history(self, history):
        import matplotlib.pyplot as plt
        import os
        figs_dir = self.config.get("evaluation", {}).get("output_dir", "outputs/figures")
        os.makedirs(figs_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss History')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_auroc'], label='Train')
        plt.plot(history['val_auroc'], label='Validation')
        plt.title('AUROC History')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "training_curves.png"))
        plt.close()

    def save_checkpoint(self, filename="best_patient_model.pt"):
        path = os.path.join(self.checkpoint_dir, filename)
        
        state = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_validation_metric': self.best_metric,
            'config': self.config,
            'pos_weight': self.pos_weight.item() if self.pos_weight is not None else None
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
        print(f"Resumed from checkpoint: {path} (Epoch {self.best_epoch})")
