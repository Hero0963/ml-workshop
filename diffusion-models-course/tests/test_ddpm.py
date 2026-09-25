# tests/test_ddpm.py
import torch

from diffusion_course.ddpm import (
    ddpm_loss,
    ddpm_sample,
    mean_from_eps,
    posterior_mean,
    predict_x0,
    q_sample,
)
from diffusion_course.models import ToyMLP
from diffusion_course.schedules import NoiseSchedule
from oracles import single_point_oracle

N = 200_000


def test_q_sample_matches_closed_form_moments() -> None:
    schedule = NoiseSchedule.create("linear", 1000)
    x0 = torch.full((N, 1), 0.7)
    t = torch.full((N,), 300)
    x_t = q_sample(schedule, x0, t, torch.randn_like(x0))
    alpha_bar = schedule.alpha_bars[300]
    assert torch.isclose(x_t.mean(), alpha_bar.sqrt() * 0.7, atol=0.01)
    assert torch.isclose(x_t.std(), (1 - alpha_bar).sqrt(), atol=0.01)


def test_many_small_steps_equal_one_big_jump() -> None:
    schedule = NoiseSchedule.create("linear", 1000)
    x = torch.full((N, 1), 0.7)
    for beta in schedule.betas[:50]:
        x = (1 - beta).sqrt() * x + beta.sqrt() * torch.randn_like(x)
    alpha_bar = schedule.alpha_bars[49]
    assert torch.isclose(x.mean(), alpha_bar.sqrt() * 0.7, atol=0.01)
    assert torch.isclose(x.std(), (1 - alpha_bar).sqrt(), atol=0.01)


def test_posterior_mean_matches_regression_on_samples() -> None:
    """E[x_{t-1} | x_t, x_0] is linear in x_t; fit it from samples and compare."""
    schedule = NoiseSchedule.create("linear", 1000)
    step = 200
    x0 = torch.full((N, 1), 0.5)
    x_prev = q_sample(schedule, x0, torch.full((N,), step - 1), torch.randn_like(x0))
    beta = schedule.betas[step]
    x_t = (1 - beta).sqrt() * x_prev + beta.sqrt() * torch.randn_like(x_prev)
    design = torch.cat([torch.ones_like(x_t), x_t], dim=1)
    intercept, slope = torch.linalg.lstsq(design, x_prev).solution.flatten()
    t = torch.tensor([step])
    predicted_at_zero = posterior_mean(schedule, x0[:1], torch.zeros(1, 1), t)
    predicted_at_one = posterior_mean(schedule, x0[:1], torch.ones(1, 1), t)
    assert torch.isclose(intercept, predicted_at_zero.squeeze(), atol=0.01)
    assert torch.isclose(
        slope, (predicted_at_one - predicted_at_zero).squeeze(), atol=0.01
    )


def test_x0_route_and_eps_route_give_the_same_mean() -> None:
    schedule = NoiseSchedule.create("cosine", 1000)
    x_t = torch.randn(16, 2)
    eps = torch.randn(16, 2)
    t = torch.randint(0, 1000, (16,))
    via_x0 = posterior_mean(schedule, predict_x0(schedule, x_t, t, eps), x_t, t)
    assert torch.allclose(via_x0, mean_from_eps(schedule, x_t, t, eps), atol=1e-4)


def test_loss_is_a_differentiable_scalar() -> None:
    schedule = NoiseSchedule.create("linear", 100)
    model = ToyMLP()
    loss = ddpm_loss(model, schedule, torch.randn(32, 2))
    loss.backward()
    assert loss.dim() == 0
    assert model.input.weight.grad is not None


def test_sampling_with_the_oracle_recovers_the_data_point() -> None:
    schedule = NoiseSchedule.create("linear", 1000)
    target = torch.tensor([[1.5, -0.5]])
    sample, trajectory = ddpm_sample(
        single_point_oracle(schedule, target), schedule, (8, 2), save_every=100
    )
    assert torch.allclose(sample, target.expand(8, 2), atol=1e-3)
    assert trajectory.shape == (11, 8, 2)
