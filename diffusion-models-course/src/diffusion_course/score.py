# src/diffusion_course/score.py
"""Score functions and Langevin dynamics -- the "score-based" view of diffusion.

The score of a density p is grad_x log p(x): an arrow at every point that says which
way the density increases fastest. Knowing only the score is already enough to sample.
"""

from collections.abc import Callable

import torch

ScoreFn = Callable[[torch.Tensor], torch.Tensor]
NoiseScoreFn = Callable[[torch.Tensor, float], torch.Tensor]


def eps_to_score(eps: torch.Tensor, one_minus_alpha_bar: torch.Tensor) -> torch.Tensor:
    """A noise predictor is a scaled score: grad log p_t(x_t) = -eps / sqrt(1 - alpha_bar_t)."""
    return -eps / one_minus_alpha_bar.sqrt()


def langevin_dynamics(
    score_fn: ScoreFn,
    x: torch.Tensor,
    step_size: float,
    num_steps: int,
    save_every: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """x <- x + step_size * score(x) + sqrt(2 * step_size) * z.

    Climb the density (the score term) while jittering (the noise term); for a small
    step and many steps the samples are distributed according to p.
    """
    trajectory = [x]
    for k in range(1, num_steps + 1):
        x = x + step_size * score_fn(x) + (2 * step_size) ** 0.5 * torch.randn_like(x)
        if save_every and k % save_every == 0:
            trajectory.append(x)
    if save_every:
        return x, torch.stack(trajectory)
    return x


def annealed_langevin_dynamics(
    score_fn: NoiseScoreFn,
    x: torch.Tensor,
    sigmas: list[float],
    steps_per_level: int,
    base_step_size: float,
) -> torch.Tensor:
    """Song & Ermon (2019): run Langevin at a decreasing sequence of noise levels.

    At a large sigma the blurred density is easy to explore; each smaller sigma refines
    the samples. The step size shrinks as sigma^2 so every level moves "the same amount"
    relative to its own scale.
    """
    smallest = sigmas[-1]
    for sigma in sigmas:
        step_size = base_step_size * (sigma / smallest) ** 2

        def score_at_level(v: torch.Tensor, level: float = sigma) -> torch.Tensor:
            return score_fn(v, level)

        x = langevin_dynamics(score_at_level, x, step_size, steps_per_level)
    return x
