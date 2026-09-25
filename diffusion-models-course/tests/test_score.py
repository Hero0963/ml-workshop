# tests/test_score.py
import torch

from diffusion_course.data import GaussianMixture
from diffusion_course.score import (
    annealed_langevin_dynamics,
    eps_to_score,
    langevin_dynamics,
)


def test_closed_form_score_matches_autograd() -> None:
    mixture = GaussianMixture.ring()
    x = (3 * torch.randn(64, 2)).requires_grad_(True)
    (grad,) = torch.autograd.grad(mixture.log_prob(x).sum(), x)
    assert torch.allclose(mixture.score(x.detach()), grad, atol=1e-4)


def test_diffused_mixture_matches_noised_samples() -> None:
    mixture = GaussianMixture.ring()
    alpha_bar = 0.3
    x0 = mixture.sample(200_000)
    x_t = alpha_bar**0.5 * x0 + (1 - alpha_bar) ** 0.5 * torch.randn_like(x0)
    expected = mixture.diffused(alpha_bar).sample(200_000)
    assert torch.allclose(x_t.pow(2).mean(0), expected.pow(2).mean(0), atol=0.02)


def test_eps_to_score_scales_and_flips() -> None:
    eps = torch.tensor([[1.0, -2.0]])
    assert torch.allclose(
        eps_to_score(eps, torch.tensor(0.25)), torch.tensor([[-2.0, 4.0]])
    )


def test_langevin_samples_a_standard_normal() -> None:
    x = torch.full((5000, 2), 4.0)
    samples = langevin_dynamics(lambda v: -v, x, step_size=0.05, num_steps=500)
    assert samples.mean().abs() < 0.1
    assert (samples.var() - 1).abs() < 0.1


def test_annealed_langevin_reaches_all_modes() -> None:
    mixture = GaussianMixture.ring(num_components=4, radius=2.0, std=0.2)

    def score(x: torch.Tensor, sigma: float) -> torch.Tensor:
        blurred = GaussianMixture(mixture.means, (mixture.std**2 + sigma**2) ** 0.5)
        return blurred.score(x)

    x = 3 * torch.randn(4000, 2)
    samples = annealed_langevin_dynamics(score, x, [2.0, 1.0, 0.5, 0.1], 100, 1e-3)
    nearest = torch.cdist(samples, mixture.means).argmin(dim=1)
    counts = torch.bincount(nearest, minlength=4).float() / len(samples)
    assert torch.all((counts - 0.25).abs() < 0.08)
