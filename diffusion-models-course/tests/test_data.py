# tests/test_data.py
import pytest
import torch

from diffusion_course.data import (
    TOY_DATASETS,
    make_toy_data,
    to_unit_range,
    toy_batches,
)


@pytest.mark.parametrize("name", TOY_DATASETS)
def test_toy_data_is_roughly_unit_scale(name: str) -> None:
    points = make_toy_data(name, 5000)
    assert points.shape == (5000, 2)
    assert 0.3 < points.std().item() < 2.5
    assert points.abs().max().item() < 3.5


def test_unknown_toy_dataset_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_toy_data("spiral", 10)


def test_toy_batches_are_fresh_and_unlabelled() -> None:
    batches = toy_batches("moons", 16)
    (a, label), (b, _) = next(batches), next(batches)
    assert label is None
    assert a.shape == (16, 2)
    assert not torch.equal(a, b)


def test_to_unit_range_clamps() -> None:
    assert torch.equal(
        to_unit_range(torch.tensor([-2.0, -1.0, 0.0, 1.0])),
        torch.tensor([0.0, 0.0, 0.5, 1.0]),
    )
