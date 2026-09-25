# src/diffusion_course/models/embeddings.py
"""How a network is told "what time it is" and "which class to draw"."""

import math

import torch
from torch import nn

TIME_SCALE = 1000.0
MAX_PERIOD = 10_000.0


class SinusoidalEmbedding(nn.Module):
    """Transformer-style sin/cos features of a scalar time t in [0, 1].

    t is first stretched to [0, 1000] so the lowest frequencies still change noticeably
    between neighbouring DDPM steps.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(MAX_PERIOD) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = TIME_SCALE * t.float()[:, None] * self.freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class TimeEmbedding(nn.Module):
    """Sinusoidal features followed by a small MLP, as in the DDPM U-Net."""

    def __init__(self, dim: int, frequency_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(frequency_dim),
            nn.Linear(frequency_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class LabelEmbedding(nn.Module):
    """One learned vector per class, plus one extra "null" class for classifier-free guidance.

    ``y=None`` means unconditional and maps every sample to the null class.
    """

    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.null_class = num_classes
        self.table = nn.Embedding(num_classes + 1, dim)

    def forward(self, y: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if y is None:
            y = torch.full(
                (batch_size,), self.null_class, device=self.table.weight.device
            )
        return self.table(y)
