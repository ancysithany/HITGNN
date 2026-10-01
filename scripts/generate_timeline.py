import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.factory import create_dataset_adapter
from src.visualization.timeline import plot_patient_timeline

def main():
    config = load_config("configs/default.yaml")
    adapter = create_dataset_adapter(config)
    cohort = adapter.prepare()
    
    # Take first patient
    patient = cohort.patients[0]
    
    os.makedirs("outputs/figures", exist_ok=True)
    out_path = "outputs/figures/patient_timeline.png"
    
    plot_patient_timeline(patient, save_path=out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
