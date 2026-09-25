# tests/test_flow_matching.py
import pytest
import torch

from diffusion_course.flow_matching import flow_matching_loss, flow_sample, interpolate


def single_point_velocity(target: torch.Tensor):
    """The exact velocity field when the data distribution is one point."""

    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        return (target - x) / (1 - t)[:, None]

    return model


def test_interpolation_endpoints() -> None:
    noise, data = torch.randn(4, 2), torch.randn(4, 2)
    assert torch.equal(interpolate(noise, data, torch.zeros(4)), noise)
    assert torch.equal(interpolate(noise, data, torch.ones(4)), data)


def test_oracle_velocity_has_zero_loss() -> None:
    target = torch.tensor([[1.0, 2.0]])
    loss = flow_matching_loss(single_point_velocity(target), target.expand(256, 2))
    assert loss.item() < 1e-8


def test_euler_with_the_oracle_lands_on_the_data_point() -> None:
    target = torch.tensor([[1.0, 2.0]])
    sample = flow_sample(single_point_velocity(target), (8, 2), num_steps=4)
    assert torch.allclose(sample, target.expand(8, 2), atol=1e-5)


def test_heun_is_exact_for_a_constant_velocity() -> None:
    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        return torch.ones_like(x)

    noise = torch.zeros(3, 2)
    sample, trajectory = flow_sample(
        model, (3, 2), num_steps=5, method="heun", noise=noise, return_trajectory=True
    )
    assert torch.allclose(sample, torch.ones(3, 2))
    assert trajectory.shape == (6, 3, 2)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        flow_sample(lambda x, t, y: x, (1, 2), method="rk4")
