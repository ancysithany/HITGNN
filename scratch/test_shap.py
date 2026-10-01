import torch
import torch.nn as nn
import shap
import numpy as np

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        
    def forward(self, x):
        # Return [B, 1] or [B] depending on test
        out = self.fc(x)
        return out.view(-1, 1)

class DummyModel1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        
    def forward(self, x):
        # Return [B] 
        out = self.fc(x)
        return out.view(-1)

print("Test [B] output (Current)")
model1d = DummyModel1D()
bg = torch.randn(32, 10)
explainer1d = shap.GradientExplainer(model1d, bg)
try:
    explainer1d.shap_values(torch.randn(1, 10))
    print("SUCCESS 1D")
except Exception as e:
    print(f"FAILED 1D: {e}")

print("Test [B, 1] output (Suggested)")
model2d = DummyModel()
bg2 = torch.randn(32, 10)
explainer2d = shap.GradientExplainer(model2d, bg2)
try:
    explainer2d.shap_values(torch.randn(1, 10))
    print("SUCCESS 2D")
except Exception as e:
    print(f"FAILED 2D: {e}")
