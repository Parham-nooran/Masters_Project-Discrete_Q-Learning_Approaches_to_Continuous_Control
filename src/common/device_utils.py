"""
Device utility for automatic TPU/GPU/CPU detection and initialization.
"""

import torch
from typing import Tuple


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


def load_checkpoint(filepath: str, device: torch.device, is_tpu: bool):
    """Load checkpoint with TPU support."""
    if is_tpu:
        return torch.load(filepath)
    else:
        return torch.load(filepath, map_location=device)