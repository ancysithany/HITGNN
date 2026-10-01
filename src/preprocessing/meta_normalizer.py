"""
src.preprocessing.meta_normalizer — Normalizes patient demographic metadata.
"""

from typing import List, Dict, Any
import numpy as np

class PatientMetaNormalizer:
    """Normalizes patient metadata using training statistics only.
    
    Fits only on the 'age' and 'sofa_score' fields.
    """
    
    def __init__(self) -> None:
        self.age_mean = 0.0
        self.age_std = 1.0
        self.sofa_mean = 0.0
        self.sofa_std = 1.0
        self.is_fitted = False
        
    def fit(self, train_records: List[Any]) -> None:
        """Fit normalization statistics on training patient records."""
        ages = []
        sofas = []
        
        for p in train_records:
            ages.append(p.metadata["age"])
            sofas.append(p.metadata["sofa_score"])
            
        if len(ages) > 0:
            self.age_mean = float(np.mean(ages))
            self.age_std = float(np.std(ages))
            if self.age_std == 0:
                self.age_std = 1.0
                
        if len(sofas) > 0:
            self.sofa_mean = float(np.mean(sofas))
            self.sofa_std = float(np.std(sofas))
            if self.sofa_std == 0:
                self.sofa_std = 1.0
                
        self.is_fitted = True
        
    def transform(self, age: float, sofa_score: float) -> tuple[float, float]:
        """Transform age and sofa_score using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("PatientMetaNormalizer must be fitted before transform.")
            
        age_norm = (age - self.age_mean) / self.age_std
        sofa_norm = (sofa_score - self.sofa_mean) / self.sofa_std
        
        return age_norm, sofa_norm
    
    def state_dict(self) -> Dict[str, float]:
        return {
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "sofa_mean": self.sofa_mean,
            "sofa_std": self.sofa_std,
            "is_fitted": self.is_fitted
        }
        
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.age_mean = state["age_mean"]
        self.age_std = state["age_std"]
        self.sofa_mean = state["sofa_mean"]
        self.sofa_std = state["sofa_std"]
        self.is_fitted = state["is_fitted"]
