# src/lm_course/utils.py
"""Small helpers shared by the labs and scripts."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


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


def notebook_logging() -> None:
    """Plain one-line log messages on stdout, so notebook outputs stay readable."""
    logger.remove()
    logger.add(sys.stdout, format="{message}")
