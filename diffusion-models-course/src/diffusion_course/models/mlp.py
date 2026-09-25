# src/diffusion_course/models/mlp.py
"""A residual MLP for 2D toy data: small enough to train in under a minute on a CPU."""

import torch
from torch import nn

from diffusion_course.models.embeddings import LabelEmbedding, TimeEmbedding


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.cond = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.fc1(self.act(self.norm(h))) + self.cond(cond)
        return h + self.fc2(self.act(z))


class ToyMLP(nn.Module):
    """model(x, t, y) -> same shape as x (noise or velocity, depending on the loss)."""

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        self.label_embed = (
            LabelEmbedding(num_classes, hidden_dim) if num_classes is not None else None
        )
        self.input = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, data_dim)
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        cond = self.time_embed(t)
        if self.label_embed is not None:
            cond = cond + self.label_embed(y, x.shape[0])
        h = self.input(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.output(h)
