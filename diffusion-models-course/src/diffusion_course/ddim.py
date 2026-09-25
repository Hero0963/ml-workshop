# src/diffusion_course/ddim.py
"""DDIM (Song, Meng & Ermon, 2020): sample a DDPM-trained model in far fewer steps.

The trained network is unchanged -- DDIM only changes how we walk back from noise.
With eta = 0 every step is deterministic, which makes it a discretisation of the
probability-flow ODE; eta = 1 recovers DDPM-like stochastic sampling.
"""

import torch

from diffusion_course.ddpm import Denoiser
from diffusion_course.schedules import NoiseSchedule


def ddim_timesteps(num_train_steps: int, num_sample_steps: int) -> list[int]:
    """Evenly spaced training steps to visit, from the noisiest down to 0."""
    steps = torch.linspace(num_train_steps - 1, 0, num_sample_steps).round().long()
    return list(dict.fromkeys(steps.tolist()))


@torch.no_grad()
def ddim_sample(
    model: Denoiser,
    schedule: NoiseSchedule,
    shape: tuple[int, ...],
    num_steps: int = 50,
    eta: float = 0.0,
    y: torch.Tensor | None = None,
    clip_x0: bool = False,
    noise: torch.Tensor | None = None,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Each step: predict x_0, then re-noise it to the next (less noisy) level.

    x_prev = sqrt(ab_prev) x0_hat + sqrt(1 - ab_prev - sigma^2) eps_hat + sigma z
    """
    device = schedule.betas.device
    x = torch.randn(shape, device=device) if noise is None else noise.to(device)
    trajectory = [x]
    steps = ddim_timesteps(schedule.num_steps, num_steps)
    for i, step in enumerate(steps):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        alpha_bar = schedule.alpha_bars[step]
        prev_step = steps[i + 1] if i + 1 < len(steps) else None
        alpha_bar_prev = (
            schedule.alpha_bars[prev_step]
            if prev_step is not None
            else x.new_tensor(1.0)
        )

        eps = model(x, schedule.timestep_input(t), y)
        x0_hat = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)
            eps = (x - alpha_bar.sqrt() * x0_hat) / (1 - alpha_bar).sqrt()

        sigma = (
            eta
            * ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
            * (1 - alpha_bar / alpha_bar_prev).sqrt()
        )
        direction = (1 - alpha_bar_prev - sigma**2).clamp(min=0).sqrt() * eps
        x = alpha_bar_prev.sqrt() * x0_hat + direction
        if eta > 0 and prev_step is not None:
            x = x + sigma * torch.randn_like(x)
        trajectory.append(x)
    if return_trajectory:
        return x, torch.stack(trajectory)
    return x
