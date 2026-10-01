"""
src.utils — Utility modules for configuration, seeding, and device management.

Imports are lazy to avoid cascading torch dependency when only config
is needed (e.g., during data generation).
"""


def __getattr__(name: str):
    """Lazy import to avoid cascading torch dependency."""
    if name == "load_config":
        from src.utils.config import load_config
        return load_config
    elif name == "set_seed":
        from src.utils.seed import set_seed
        return set_seed
    elif name == "get_device":
        from src.utils.device import get_device
        return get_device
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")


__all__ = ["load_config", "set_seed", "get_device"]
