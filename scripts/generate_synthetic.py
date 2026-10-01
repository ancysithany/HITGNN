import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.synthetic_generator import SyntheticCohortGenerator, patient_level_split, save_cohort
from src.data.vocab import ConceptVocabulary

def main():
    config = load_config('configs/default.yaml')
    seed = config.get('seed', 42)
    set_seed(seed)
    
    vocab = ConceptVocabulary.build_default()
    gen = SyntheticCohortGenerator(config, vocab, seed=seed)
    patients = gen.generate()
    
    os.makedirs('data/synthetic', exist_ok=True)
    save_cohort(patients, 'data/synthetic')
    
    split_ratios = config.get('training', {}).get('split_ratios', {'train': 0.70, 'val': 0.15, 'test': 0.15})
    ids = [p.patient_id for p in patients]
    labels = [p.mortality for p in patients]
    splits = patient_level_split(ids, split_ratios, seed=seed, labels=labels)
    
    with open('data/synthetic/splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
        
    print("Saved generated synthetic dataset and splits.")

if __name__ == "__main__":
    main()
