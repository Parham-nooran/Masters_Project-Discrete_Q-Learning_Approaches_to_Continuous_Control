"""
Device utility for automatic TPU/GPU/CPU detection and initialization.
"""

import torch
from typing import Tuple


def get_device() -> Tuple[torch.device, bool, bool]:
    """
    Detect and return the best available device (TPU > GPU > CPU).

    Returns:
        Tuple of (device, is_tpu, use_amp)
    """

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"Using TPU: {device}")
        return device, True, False
    except ImportError:
        pass
    except Exception as e:
        print(f"TPU initialization failed: {e}")


    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device, False, True


    device = torch.device("cpu")
    print("Using CPU")
    return device, False, False


def optimizer_step(optimizer, is_tpu: bool):
    """Perform optimizer step with TPU support."""
    if is_tpu:
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()


def mark_step(is_tpu: bool):
    """Mark step for TPU synchronization."""
    if is_tpu:
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def save_checkpoint(checkpoint_dict: dict, filepath: str, is_tpu: bool):
    """Save checkpoint with TPU support."""
    if is_tpu:
        import torch_xla.core.xla_model as xm
        xm.save(checkpoint_dict, filepath)
    else:
        torch.save(checkpoint_dict, filepath)


def load_checkpoint(filepath: str, device: torch.device, is_tpu: bool):
    """Load checkpoint with TPU support."""
    if is_tpu:
        return torch.load(filepath)
    else:
        return torch.load(filepath, map_location=device)