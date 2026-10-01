# HITGNN Demo

**Hierarchical Temporal Event Graph Neural Network
for Missingness-Aware Sepsis Mortality Prediction**



## Overview

HITGNN represents ICU patients as temporal event graphs rather than flat feature
vectors. Each observed clinical event (lab result, vital sign, medication, etc.)
becomes a graph node, connected by four edge types that encode temporal
adjacency, concept recurrence, semantic relationships, and cross-modality links.

**Prediction task:** Given ICU data available up to Sepsis-3 recognition time
t₀, predict in-hospital mortality (binary: dies in hospital vs. discharged
alive).

### Five Core Ideas

1. **Temporal Clinical Event Graph** — per-patient directed attributed graph
2. **Missingness-Aware Biomarker Encoder** — measurement patterns as signal
3. **Temporal Edge-Aware GAT (TE-GAT)** — time participates directly in attention
4. **Hierarchical Patient Representation** — event → type → patient pooling
5. **Inductive Population Graph** — k-NN similarity with frozen training bank

## Project Structure

```
hitgnn-demo/
├── configs/
│   └── default.yaml              # All hyperparameters and settings
├── data/
│   ├── raw/                      # Real data (MIMIC-IV/eICU) — future
│   ├── processed/                # Preprocessed data artifacts
│   └── synthetic/                # Generated synthetic cohort
├── src/
│   ├── data/                     # Data generation, vocab, terminology
│   ├── preprocessing/            # Normalization, temporal binning
│   ├── graph/                    # Graph construction, population graph
│   ├── models/                   # TE-GAT, pooling, encoder, full model
│   ├── explainability/           # 4-level associational attribution
│   ├── evaluation/               # Metrics, plotting
│   └── utils/                    # Config, seeding, device utilities
├── dashboard/                    # Streamlit interactive demo
├── scripts/                      # CLI entry points
├── tests/                        # Unit and integration tests
├── notebooks/                    # Exploration notebooks
├── docs/                         # Documentation
└── outputs/
    ├── checkpoints/              # Saved model weights
    ├── figures/                  # ROC, PR, calibration plots
    ├── metrics/                  # Evaluation results
    └── explanations/             # Attribution outputs
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python scripts/generate_data.py --config configs/default.yaml

# 3. Train model
python scripts/train.py --config configs/default.yaml

# 4. Evaluate
python scripts/evaluate.py --config configs/default.yaml

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## Locked Design Decisions

| Decision | Specification |
|---|---|
| Graph format | Homogeneous PyG `Data` with type attributes |
| Prediction head | Raw logits → `BCEWithLogitsLoss`; sigmoid at inference only |
| TE-GAT attention | `e = LeakyReLU(a^T [Wh_v ∥ Wh_u ∥ φ(Δt)])` — time in logit |
| Missingness | `[x*m, m, log(1+Δt), n]` — no mean imputation |
| Population graph | Inductive: test/val → train only; no label leakage |
| Explanations | Associational attribution only — never causal claims |
| UI terminology | "Predicted mortality probability" — never "confidence" |

## License

Research demonstration. See disclaimer above.
