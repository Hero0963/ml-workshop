# src/diffusion_course/flow_matching.py
"""Flow matching / rectified flow with the straight-line (optimal-transport) path.

Time runs from t = 0 (pure noise) to t = 1 (data), following Lipman et al. (2022) and
the "Flow Matching Guide and Code" (2024). Beware: DDPM calls the *data* x_0, and SD3
runs time the other way (t = 0 is data), so the code names tensors ``noise`` and
``data`` instead of x_0 / x_1.

    x_t = t * data + (1 - t) * noise        target velocity: data - noise
"""

import torch
import torch.nn.functional as F

from diffusion_course.ddpm import Denoiser
from diffusion_course.utils import broadcast_to


def interpolate(
    noise: torch.Tensor, data: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    t = broadcast_to(t, data)
    return t * data + (1 - t) * noise


def flow_matching_loss(
    model: Denoiser,
    data: torch.Tensor,
    y: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regress the velocity of a random straight line from noise to data.

    Passing ``noise`` fixes the pairing, which is what reflow (rectified flow) needs.
    """
    noise = torch.randn_like(data) if noise is None else noise
    t = torch.rand(data.shape[0], device=data.device)
    x_t = interpolate(noise, data, t)
    return F.mse_loss(model(x_t, t, y), data - noise)


@torch.no_grad()
def flow_sample(
    model: Denoiser,
    shape: tuple[int, ...],
    num_steps: int = 50,
    method: str = "euler",
    y: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate dx/dt = v(x, t) from t = 0 to 1 with Euler or Heun (2nd order) steps."""
    if method not in ("euler", "heun"):
        raise ValueError(f"unknown method {method!r}; choose 'euler' or 'heun'")
    x = torch.randn(shape, device=device) if noise is None else noise.to(device)
    trajectory = [x]
    times = torch.linspace(0, 1, num_steps + 1, device=x.device)
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = t1 - t0
        v0 = model(x, t0.expand(shape[0]), y)
        if method == "heun":
            x_euler = x + dt * v0
            v1 = model(x_euler, t1.expand(shape[0]), y)
            x = x + dt * (v0 + v1) / 2
        else:
            x = x + dt * v0
        trajectory.append(x)
    if return_trajectory:
        return x, torch.stack(trajectory)
    return x
