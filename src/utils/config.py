"""
src.utils.config — Configuration loading and validation.

Loads YAML configuration files and provides typed access to all
HITGNN hyperparameters, paths, and settings.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dictionary.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file, relative to the project root
        or as an absolute path.

    Returns
    -------
    Dict[str, Any]
        Nested dictionary containing all configuration parameters.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested configuration value.

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration dictionary.
    *keys : str
        Sequence of keys to traverse (e.g., ``"te_gat", "n_heads"``).
    default : Any
        Value to return if the key path does not exist.

    Returns
    -------
    Any
        The configuration value, or ``default`` if not found.
    """
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current
