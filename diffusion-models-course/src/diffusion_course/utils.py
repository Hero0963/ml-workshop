# src/diffusion_course/utils.py
"""Small helpers shared by the labs and scripts."""

import random

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def broadcast_to(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample vector of shape (B,) so it broadcasts against x of shape (B, ...)."""
    return values.reshape(-1, *([1] * (x.dim() - 1)))
