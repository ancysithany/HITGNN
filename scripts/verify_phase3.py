"""
Phase 3 verification script — Biomarker State + Missingness Encoder.

Demonstrates the full preprocessing pipeline:
    1. Load synthetic cohort + splits
    2. Fit biomarker normalizer on TRAIN ONLY
    3. Build BiomarkerState for all patients
    4. Instantiate MissingnessEncoder
    5. Run a batch through the encoder
    6. Print tensor shapes
    7. Save diagnostic outputs

Usage:
    python scripts/verify_phase3.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.data.vocab import ConceptVocabulary
from src.data.synthetic_generator import (
    SyntheticCohortGenerator, PatientRecord,
    patient_level_split,
)
from src.preprocessing.temporal_binner import TemporalBinner, BiomarkerState
from src.preprocessing.normalizer import FeatureNormalizer
from src.models.missingness_encoder import MissingnessEncoder


def main():
    parser = argparse.ArgumentParser(description="Phase 3 verification")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    biomarker_names = config.get("biomarkers", [
        "PCT", "CRP", "Lactate", "WBC", "Creatinine", "Bilirubin", "Platelets"
    ])
    n_bins = config.get("temporal", {}).get("n_bins", 6)

    # ---- 1. Load/generate cohort ----
    print("=" * 60)
    print("Phase 3 Verification: Biomarker State + Missingness Encoder")
    print("=" * 60)

    print("\n1. Generating synthetic cohort...")
    vocab = ConceptVocabulary.build_default()
    gen = SyntheticCohortGenerator(config, vocab, seed=seed)
    patients = gen.generate()

    split_ratios = config.get("training", {}).get(
        "split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    ids = [p.patient_id for p in patients]
    labels = [p.mortality for p in patients]
    splits = patient_level_split(ids, split_ratios, seed=seed, labels=labels)
    pid_to_patient = {p.patient_id: p for p in patients}

    train_patients = [pid_to_patient[pid] for pid in splits["train"]]
    val_patients = [pid_to_patient[pid] for pid in splits["val"]]
    test_patients = [pid_to_patient[pid] for pid in splits["test"]]

    print(f"   Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

    # ---- 2. Fit normalizer on TRAIN ONLY ----
    print("\n2. Fitting biomarker normalizer on TRAIN split only...")
    normalizer = FeatureNormalizer(method="zscore")
    for bm in biomarker_names:
        train_vals = []
        for p in train_patients:
            for e in p.events:
                if e.event_type == "LAB" and e.concept_name == bm and e.value is not None:
                    train_vals.append(e.value)
        normalizer.fit(bm, np.array(train_vals) if train_vals else np.array([]))

    print("   Normalization statistics (train-only):")
    for bm in biomarker_names:
        if bm in normalizer.stats:
            mean, std = normalizer.stats[bm]
            print(f"     {bm:15s}  mean={mean:8.3f}  std={std:8.3f}")

    # Save normalizer
    norm_path = "data/processed/normalizer_stats.json"
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    normalizer.save(norm_path)
    print(f"   Saved: {norm_path}")

    # ---- 3. Build BiomarkerStates ----
    print(f"\n3. Building BiomarkerState (K={n_bins}, B={len(biomarker_names)})...")
    binner = TemporalBinner(n_bins=n_bins, delta_t_max_hours=200.0)

    def build_states(patient_list):
        states = []
        for p in patient_list:
            timestamps = [e.timestamp for e in p.events if e.event_type == "LAB"]
            if timestamps:
                window_start = min(timestamps)
            else:
                window_start = -24.0  # Default
            state = binner.build_biomarker_state(
                p.events, biomarker_names,
                window_start=window_start, window_end=0.0,
                normalizer=normalizer,
            )
            state.patient_id = p.patient_id
            states.append(state)
        return states

    train_states = build_states(train_patients)
    val_states = build_states(val_patients)
    test_states = build_states(test_patients)

    print(f"   Built {len(train_states)} train, {len(val_states)} val, {len(test_states)} test states")
    print(f"   BiomarkerState shape: values=[{n_bins}, {len(biomarker_names)}]")

    # ---- 4. Stack into batch tensors ----
    print("\n4. Converting to PyTorch tensors...")

    def states_to_tensor(states):
        """Stack BiomarkerStates into [batch, K, B, 4] input tensor."""
        batch = []
        for s in states:
            # Channel order: [x*mask, mask, log(1+delta_t), never_observed]
            # values already has x * mask applied
            slot = np.stack([
                s.values,          # x_normalized * mask
                s.masks,           # mask
                s.delta_times,     # log(1 + delta_t)
                s.never_observed,  # never_observed
            ], axis=-1)  # [K, B, 4]
            batch.append(slot)
        return torch.tensor(np.stack(batch, axis=0), dtype=torch.float32)

    train_tensor = states_to_tensor(train_states)
    val_tensor = states_to_tensor(val_states)

    print(f"   Train input tensor: {train_tensor.shape}")
    print(f"   Val input tensor:   {val_tensor.shape}")

    # ---- 5. Instantiate and run MissingnessEncoder ----
    print("\n5. Running MissingnessEncoder...")
    miss_cfg = config.get("missingness_encoder", {})
    encoder = MissingnessEncoder(
        input_dim=miss_cfg.get("input_dim", 4),
        hidden_dim=miss_cfg.get("hidden_dim", 64),
        output_dim=miss_cfg.get("output_dim", 64),
        n_biomarkers=len(biomarker_names),
        n_bins=n_bins,
        output_activation=miss_cfg.get("output_activation", None),
    )
    encoder.eval()

    with torch.no_grad():
        # Process in mini-batches to avoid memory issues
        batch_size = 32
        mini_batch = train_tensor[:batch_size]
        slot_emb, z_miss = encoder(mini_batch)

    print(f"   Encoder input:       {mini_batch.shape}")
    print(f"   Slot embeddings:     {slot_emb.shape}")
    print(f"   Patient summary:     {z_miss.shape}")

    # ---- 6. Diagnostic output ----
    print("\n6. Saving diagnostic outputs...")
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Example missingness state CSV
    example_state = train_states[0]
    rows = []
    for k in range(n_bins):
        for bi, bm in enumerate(biomarker_names):
            rows.append({
                "bin": k,
                "biomarker": bm,
                "raw_value": round(float(example_state.raw_values[k, bi]), 4),
                "normalized_value": round(float(example_state.values[k, bi]), 4),
                "mask": int(example_state.masks[k, bi]),
                "delta_t_hours": round(float(np.exp(example_state.delta_times[k, bi]) - 1.0), 2),
                "log_delta_t": round(float(example_state.delta_times[k, bi]), 4),
                "never_observed": int(example_state.never_observed[k, bi]),
            })
    diag_df = pd.DataFrame(rows)
    diag_path = "outputs/metrics/example_missingness_state.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"   Saved: {diag_path}")

    # Missingness heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Mask heatmap
        ax = axes[0]
        im = ax.imshow(example_state.masks.T, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(biomarker_names)))
        ax.set_yticklabels(biomarker_names, fontsize=9)
        ax.set_xlabel("Temporal Bin")
        ax.set_title(f"Observation Mask - {example_state.patient_id}")
        plt.colorbar(im, ax=ax, label="Observed")

        # Never-observed heatmap
        ax = axes[1]
        im = ax.imshow(example_state.never_observed.T, aspect="auto",
                        cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(biomarker_names)))
        ax.set_yticklabels(biomarker_names, fontsize=9)
        ax.set_xlabel("Temporal Bin")
        ax.set_title(f"Never-Observed - {example_state.patient_id}")
        plt.colorbar(im, ax=ax, label="Never Observed")

        plt.tight_layout()
        heatmap_path = "outputs/figures/missingness_mask_example.png"
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {heatmap_path}")
    except Exception as e:
        print(f"   Heatmap generation skipped: {e}")

    # ---- 7. Inspect specific examples ----
    print("\n7. Inspecting examples...")

    # Find a patient with never-observed PCT
    for s in train_states:
        pct_idx = biomarker_names.index("PCT")
        if s.never_observed[:, pct_idx].sum() == n_bins:
            pid = s.patient_id
            print(f"\n   Never-observed PCT example: {pid}")
            print(f"     PCT mask:          {s.masks[:, pct_idx].tolist()}")
            print(f"     PCT never_observed: {s.never_observed[:, pct_idx].tolist()}")
            print(f"     PCT values:        {s.values[:, pct_idx].tolist()}")
            dt_hours = [round(float(np.exp(v) - 1.0), 1) for v in s.delta_times[:, pct_idx]]
            print(f"     PCT delta_t(hours): {dt_hours}")
            break

    # Find a patient with repeated Lactate
    for s in train_states:
        lac_idx = biomarker_names.index("Lactate")
        n_obs = int(s.masks[:, lac_idx].sum())
        if n_obs >= 2:
            pid = s.patient_id
            print(f"\n   Repeated Lactate example: {pid} ({n_obs} observations)")
            print(f"     Lactate mask:      {s.masks[:, lac_idx].tolist()}")
            print(f"     Lactate raw_val:   {[round(float(v), 2) for v in s.raw_values[:, lac_idx]]}")
            print(f"     Lactate norm_val:  {[round(float(v), 2) for v in s.values[:, lac_idx]]}")
            dt_hours = [round(float(np.exp(v) - 1.0), 1) for v in s.delta_times[:, lac_idx]]
            print(f"     Lactate delta_t:   {dt_hours}")
            print(f"     Lactate never_obs: {s.never_observed[:, lac_idx].tolist()}")
            break

    print("\n" + "=" * 60)
    print("Phase 3 verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
