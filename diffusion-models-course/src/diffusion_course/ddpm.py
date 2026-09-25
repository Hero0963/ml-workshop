# src/diffusion_course/ddpm.py
"""DDPM (Ho et al., 2020): forward noising, the simple epsilon-prediction loss, and
ancestral sampling.

Every model in the course is called as ``model(x, t, y)`` where ``t`` is a float in
[0, 1) (see ``NoiseSchedule.timestep_input``) and ``y`` is an optional class label.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from diffusion_course.schedules import NoiseSchedule
from diffusion_course.utils import broadcast_to

Denoiser = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def q_sample(
    schedule: NoiseSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Jump straight to step t: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps."""
    alpha_bar = broadcast_to(schedule.alpha_bars[t], x0)
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise


def ddpm_loss(
    model: Denoiser,
    schedule: NoiseSchedule,
    x0: torch.Tensor,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_simple: pick a random step, add noise, ask the model which noise was added."""
    t = torch.randint(0, schedule.num_steps, (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(schedule, x0, t, noise)
    return F.mse_loss(model(x_t, schedule.timestep_input(t), y), noise)


def predict_x0(
    schedule: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """Invert q_sample: if eps were the true noise, this would be the clean sample."""
    alpha_bar = broadcast_to(schedule.alpha_bars[t], x_t)
    return (x_t - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()


def posterior_mean(
    schedule: NoiseSchedule, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Mean of q(x_{t-1} | x_t, x_0): a weighted average of the clean and the noisy sample."""
    alpha = broadcast_to(schedule.alphas[t], x_t)
    alpha_bar = broadcast_to(schedule.alpha_bars[t], x_t)
    alpha_bar_prev = broadcast_to(schedule.alpha_bars_prev[t], x_t)
    beta = broadcast_to(schedule.betas[t], x_t)
    coef_x0 = alpha_bar_prev.sqrt() * beta / (1 - alpha_bar)
    coef_xt = alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar)
    return coef_x0 * x0 + coef_xt * x_t


def mean_from_eps(
    schedule: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """The same posterior mean written the way the DDPM paper's Algorithm 2 does."""
    alpha = broadcast_to(schedule.alphas[t], x_t)
    alpha_bar = broadcast_to(schedule.alpha_bars[t], x_t)
    beta = broadcast_to(schedule.betas[t], x_t)
    return (x_t - beta / (1 - alpha_bar).sqrt() * eps) / alpha.sqrt()


@torch.no_grad()
def ddpm_sample(
    model: Denoiser,
    schedule: NoiseSchedule,
    shape: tuple[int, ...],
    y: torch.Tensor | None = None,
    variance: str = "beta",
    clip_x0: bool = False,
    noise: torch.Tensor | None = None,
    save_every: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Ancestral sampling (Algorithm 2): walk all T steps from pure noise back to data.

    variance: "beta" uses sigma_t^2 = beta_t, "posterior" uses the posterior variance;
        Ho et al. report both work about equally well.
    clip_x0: clamp the predicted clean sample to [-1, 1] before taking the step -- a
        standard trick for images, meaningless for the unbounded 2D toy data.
    save_every: also return the intermediate samples every that many steps, stacked
        into a tensor of shape (num_saved, *shape).
    """
    device = schedule.betas.device
    x = torch.randn(shape, device=device) if noise is None else noise.to(device)
    trajectory = [x]
    for i in reversed(range(schedule.num_steps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, schedule.timestep_input(t), y)
        x0_hat = predict_x0(schedule, x, t, eps)
        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)
        mean = posterior_mean(schedule, x0_hat, x, t)
        if i > 0:
            var = (
                schedule.betas[i]
                if variance == "beta"
                else schedule.posterior_variance[i]
            )
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean
        if save_every and i % save_every == 0:
            trajectory.append(x)
    if save_every:
        return x, torch.stack(trajectory)
    return x
