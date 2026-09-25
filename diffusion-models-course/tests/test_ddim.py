# tests/test_ddim.py
import torch

from diffusion_course.ddim import ddim_sample, ddim_timesteps
from diffusion_course.schedules import NoiseSchedule
from oracles import single_point_oracle


def test_timesteps_run_from_noisiest_to_zero() -> None:
    steps = ddim_timesteps(1000, 50)
    assert len(steps) == 50
    assert steps[0] == 999
    assert steps[-1] == 0
    assert steps == sorted(steps, reverse=True)


def test_ten_step_ddim_recovers_the_data_point() -> None:
    schedule = NoiseSchedule.create("linear", 1000)
    target = torch.tensor([[1.5, -0.5]])
    sample = ddim_sample(
        single_point_oracle(schedule, target), schedule, (8, 2), num_steps=10
    )
    assert torch.allclose(sample, target.expand(8, 2), atol=1e-3)


def test_eta_zero_is_deterministic() -> None:
    schedule = NoiseSchedule.create("linear", 1000)

    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        return 0.3 * x

    noise = torch.randn(4, 2)
    first = ddim_sample(model, schedule, (4, 2), num_steps=20, noise=noise)
    second = ddim_sample(model, schedule, (4, 2), num_steps=20, noise=noise)
    assert torch.equal(first, second)


def test_trajectory_has_one_entry_per_step_plus_start() -> None:
    schedule = NoiseSchedule.create("linear", 1000)

    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        return torch.zeros_like(x)

    _, trajectory = ddim_sample(
        model, schedule, (4, 2), num_steps=5, return_trajectory=True
    )
    assert trajectory.shape == (6, 4, 2)
