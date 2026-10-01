"""
src.utils.device — Device selection with CUDA fallback to CPU.
"""

import torch


def get_device(device_config: str = "auto") -> torch.device:
    """Resolve the compute device from configuration.

    Parameters
    ----------
    device_config : str
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
        When ``"auto"``, selects CUDA if available, otherwise CPU.

    Returns
    -------
    torch.device
        The resolved PyTorch device.
    """
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)
