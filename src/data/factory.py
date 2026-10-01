"""
src.data.factory — Factory for creating dataset adapters.
"""
from src.data.adapters.base import BaseDatasetAdapter
from src.data.adapters.synthetic import SyntheticDatasetAdapter
from src.data.adapters.mimic import MIMICDatasetAdapter

def create_dataset_adapter(config: dict) -> BaseDatasetAdapter:
    """Instantiate the correct dataset adapter based on configuration."""
    dataset_config = config.get("dataset", {})
    adapter_name = dataset_config.get("adapter", "synthetic")
    
    if adapter_name == "synthetic":
        root_dir = dataset_config.get("root", "data/synthetic")
        return SyntheticDatasetAdapter(root_dir=root_dir)
    elif adapter_name == "mimic":
        root_dir = dataset_config.get("root", "data/mimic")
        return MIMICDatasetAdapter(root_dir=root_dir)
    else:
        raise ValueError(f"Unknown dataset adapter: {adapter_name}")
