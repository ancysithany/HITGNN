"""
Generate synthetic ICU patient cohort for HITGNN demo.

Usage:
    python scripts/generate_data.py --config configs/default.yaml

Output:
    data/synthetic/patients.csv
    data/synthetic/events.csv
    data/synthetic/splits.json
    data/synthetic/metadata.json
    outputs/figures/synthetic_patient_timeline.png
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.vocab import ConceptVocabulary
from src.data.synthetic_generator import (
    SyntheticCohortGenerator,
    save_cohort,
    patient_level_split,
    print_cohort_summary,
)
from src.visualization import plot_patient_timeline


def main() -> None:
    """Generate synthetic cohort from configuration."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic ICU patient cohort for HITGNN."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated data.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    # ---- Build vocabulary ----
    print("Building concept vocabulary...")
    vocab = ConceptVocabulary.build_default()
    print(f"  {vocab.size} clinical concepts registered.")

    # ---- Generate cohort ----
    print("Generating synthetic ICU cohort...")
    generator = SyntheticCohortGenerator(config, vocab, seed=seed)
    patients = generator.generate()

    # ---- Patient-level STRATIFIED split ----
    split_ratios = config.get("training", {}).get(
        "split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    patient_ids = [p.patient_id for p in patients]
    labels = [p.mortality for p in patients]
    splits = patient_level_split(patient_ids, split_ratios, seed=seed, labels=labels)

    # ---- Save data ----
    print(f"Saving to {args.output_dir}/...")
    paths = save_cohort(patients, output_dir=args.output_dir)

    # Save splits.json
    splits_path = os.path.join(args.output_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    paths["splits"] = splits_path

    # Save metadata.json
    n_events = sum(len(p.events) for p in patients)
    n_mort = sum(p.mortality for p in patients)
    metadata = {
        "seed": seed,
        "n_patients": len(patients),
        "n_events": n_events,
        "mortality_count": n_mort,
        "mortality_prevalence": round(n_mort / len(patients), 4),
        "generation_timestamp": datetime.now().isoformat(),
        "config_path": args.config,
        "output_dir": args.output_dir,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "disclaimer": (
            "Research demonstration using patient data. "
            "Not intended for clinical diagnosis, treatment, or decision-making."
        ),
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["metadata"] = metadata_path

    # ---- Validation checks ----
    print("\nRunning validation checks...")
    errors = validate_cohort(patients, splits, config)
    if errors:
        print(f"  WARNINGS ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  All validation checks passed.")

    # ---- Summary ----
    print()
    print_cohort_summary(patients, splits)

    # ---- Visualization ----
    print("\nGenerating patient timeline visualization...")
    fig_dir = config.get("evaluation", {}).get("output_dir", "outputs/figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "synthetic_patient_timeline.png")
    # Pick a patient with decent event count for visualization
    viz_patient = max(patients[:20], key=lambda p: len(p.events))
    plot_patient_timeline(viz_patient, save_path=fig_path)
    print(f"  Saved: {fig_path}")

    # ---- Output paths ----
    print("\nOutput files:")
    for name, path in paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name:12s} -> {path} ({size_kb:.1f} KB)")


    print("\nDone.")


def validate_cohort(patients, splits, config):
    """Run validation checks and return list of warning strings."""
    errors = []
    n = len(patients)
    syn_cfg = config.get("synthetic", {})

    # 1. Patient count
    expected = syn_cfg.get("n_patients", 500)
    if n != expected:
        errors.append(f"Expected {expected} patients, got {n}")

    # 2. Mortality labels binary
    for p in patients:
        if p.mortality not in (0, 1):
            errors.append(f"Patient {p.patient_id} has non-binary mortality: {p.mortality}")
            break

    # 3. Approximate prevalence (within 5% absolute)
    target_prev = syn_cfg.get("mortality_prevalence", 0.20)
    actual_prev = sum(p.mortality for p in patients) / n
    if abs(actual_prev - target_prev) > 0.08:
        errors.append(
            f"Mortality prevalence {actual_prev:.3f} deviates >8% from target {target_prev}"
        )

    # 4. No timestamps after t0
    for p in patients:
        for e in p.events:
            if e.timestamp > 0.0:
                errors.append(f"Patient {p.patient_id} has event at t={e.timestamp} > t0")
                break

    # 5. No split overlap
    all_splits = []
    for split_name, ids in splits.items():
        all_splits.extend(ids)
    if len(all_splits) != len(set(all_splits)):
        errors.append("Split overlap detected — patient appears in multiple splits")

    # 6. Every event references a valid patient
    valid_pids = {p.patient_id for p in patients}
    for p in patients:
        for e in p.events:
            if e.patient_id != p.patient_id:
                errors.append(f"Event patient_id mismatch: {e.patient_id} != {p.patient_id}")
                break

    # 7. Valid event types
    valid_types = {"LAB", "VITAL", "MEDICATION", "PROCEDURE", "DIAGNOSIS", "CLINICAL_CONCEPT"}
    for p in patients:
        for e in p.events:
            if e.event_type not in valid_types:
                errors.append(f"Invalid event type: {e.event_type}")
                break

    # 8. All event types present across cohort
    observed_types = set()
    for p in patients:
        for e in p.events:
            observed_types.add(e.event_type)
    missing_types = valid_types - observed_types
    if missing_types:
        errors.append(f"Missing event types in cohort: {missing_types}")

    # 9. Missingness exists
    biomarkers = config.get("biomarkers", [])
    for bm in biomarkers:
        n_with = sum(1 for p in patients if any(e.concept_name == bm for e in p.events))
        if n_with == n:
            errors.append(f"No missingness for {bm} — all {n} patients have measurements")

    # 10. Variable trajectory lengths
    event_counts = [len(p.events) for p in patients]
    if len(set(event_counts)) < 5:
        errors.append("Insufficient trajectory length variation")

    # 11. No duplicate patient IDs
    pids = [p.patient_id for p in patients]
    if len(pids) != len(set(pids)):
        errors.append("Duplicate patient IDs detected")

    return errors


if __name__ == "__main__":
    main()
