# src/diffusion_course/schedules.py
"""Noise schedules for the discrete-time (DDPM) forward process.

Index convention: timestep index ``i`` runs 0..T-1 and corresponds to the paper's
t = i + 1. So ``alpha_bars[0]`` is already slightly noisy and ``alpha_bars[T - 1]`` is
(almost) pure noise.
"""

import math

import torch

DEFAULT_NUM_STEPS = 1000
LINEAR_BETA_START = 1e-4
LINEAR_BETA_END = 0.02
COSINE_OFFSET = 0.008
MAX_BETA = 0.999


def linear_betas(
    num_steps: int = DEFAULT_NUM_STEPS,
    beta_start: float = LINEAR_BETA_START,
    beta_end: float = LINEAR_BETA_END,
) -> torch.Tensor:
    """Ho et al. (2020): betas grow linearly from 1e-4 to 0.02 over 1000 steps."""
    scale = DEFAULT_NUM_STEPS / num_steps
    return torch.linspace(beta_start * scale, beta_end * scale, num_steps)


def cosine_betas(
    num_steps: int = DEFAULT_NUM_STEPS, offset: float = COSINE_OFFSET
) -> torch.Tensor:
    """Nichol & Dhariwal (2021): alpha_bar follows a squared cosine, so information is
    destroyed more evenly than with the linear schedule."""

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.cos((t / num_steps + offset) / (1 + offset) * math.pi / 2) ** 2

    t = torch.arange(num_steps + 1, dtype=torch.float64)
    alpha_bars = f(t) / f(t[:1])
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(max=MAX_BETA).float()


class NoiseSchedule:
    """Every per-timestep constant the DDPM equations need, precomputed once."""

    def __init__(self, betas: torch.Tensor) -> None:
        self.betas = betas.float()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([torch.ones(1), self.alpha_bars[:-1]])
        # Variance of q(x_{t-1} | x_t, x_0): the "posterior" the model learns to imitate.
        self.posterior_variance = (
            self.betas * (1 - self.alpha_bars_prev) / (1 - self.alpha_bars)
        )

    @classmethod
    def create(
        cls, kind: str = "linear", num_steps: int = DEFAULT_NUM_STEPS
    ) -> "NoiseSchedule":
        makers = {"linear": linear_betas, "cosine": cosine_betas}
        if kind not in makers:
            raise ValueError(f"unknown schedule {kind!r}; choose from {list(makers)}")
        return cls(makers[kind](num_steps))

    @property
    def num_steps(self) -> int:
        return self.betas.shape[0]

    @property
    def snr(self) -> torch.Tensor:
        """Signal-to-noise ratio alpha_bar / (1 - alpha_bar) at every step."""
        return self.alpha_bars / (1 - self.alpha_bars)

    def to(self, device: torch.device | str) -> "NoiseSchedule":
        moved = NoiseSchedule.__new__(NoiseSchedule)
        for name, value in vars(self).items():
            setattr(moved, name, value.to(device))
        return moved

    def timestep_input(self, t: torch.Tensor) -> torch.Tensor:
        """Map integer steps 0..T-1 onto [0, 1), the time input every model in this course takes."""
        return t.float() / self.num_steps
