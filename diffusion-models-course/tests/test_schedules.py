# tests/test_schedules.py
import pytest
import torch

from diffusion_course.schedules import (
    MAX_BETA,
    NoiseSchedule,
    cosine_betas,
    linear_betas,
)


def test_linear_betas_match_ddpm_paper() -> None:
    betas = linear_betas(1000)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_alpha_bar_decays_from_signal_to_noise(kind: str) -> None:
    schedule = NoiseSchedule.create(kind, 1000)
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    assert schedule.alpha_bars[0] > 0.99
    assert schedule.alpha_bars[-1] < 1e-3
    assert torch.all(schedule.snr[1:] < schedule.snr[:-1])


def test_cosine_betas_are_capped() -> None:
    float32_rounding = 1e-6
    assert cosine_betas(1000).max().item() <= MAX_BETA + float32_rounding


def test_first_posterior_variance_is_zero() -> None:
    schedule = NoiseSchedule.create("linear", 100)
    assert schedule.posterior_variance[0].item() == 0.0
    assert torch.all(schedule.posterior_variance[1:] <= schedule.betas[1:])


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError):
        NoiseSchedule.create("sigmoid")


def test_timestep_input_is_in_unit_interval() -> None:
    schedule = NoiseSchedule.create("linear", 1000)
    t = schedule.timestep_input(torch.tensor([0, 999]))
    assert t[0].item() == 0.0
    assert t[1].item() < 1.0
