# src/diffusion_course/guidance.py
"""Classifier-free guidance (Ho & Salimans, 2022).

Train one network that is sometimes told the label and sometimes not (the "null" label).
At sampling time, push the prediction away from the unconditional one:

    guided = uncond + scale * (cond - uncond)

scale = 1 is the plain conditional model, scale = 0 is unconditional, scale > 1 trades
diversity for fidelity. Works the same for noise (DDPM) and velocity (flow matching)
predictions, because both are linear in the score.
"""

import torch
from torch import nn


def drop_labels(y: torch.Tensor, drop_prob: float, null_class: int) -> torch.Tensor:
    """During training, replace a random fraction of labels with the null label."""
    drop = torch.rand(y.shape, device=y.device) < drop_prob
    return torch.where(drop, torch.full_like(y, null_class), y)


class ClassifierFreeGuidance(nn.Module):
    """Wrap a conditional model so any sampler in the course can use guidance unchanged."""

    def __init__(
        self, model: nn.Module, guidance_scale: float, null_class: int
    ) -> None:
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale
        self.null_class = null_class

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("classifier-free guidance needs class labels")
        null = torch.full_like(y, self.null_class)
        both = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
        cond, uncond = both.chunk(2)
        return uncond + self.guidance_scale * (cond - uncond)
