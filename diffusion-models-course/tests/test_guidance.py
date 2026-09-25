# tests/test_guidance.py
import pytest
import torch

from diffusion_course.guidance import ClassifierFreeGuidance, drop_labels
from diffusion_course.models import ToyMLP

NULL = 3


def test_drop_labels_extremes() -> None:
    y = torch.tensor([0, 1, 2, 1])
    assert torch.equal(drop_labels(y, 0.0, NULL), y)
    assert torch.equal(drop_labels(y, 1.0, NULL), torch.full_like(y, NULL))


@pytest.fixture
def model() -> ToyMLP:
    return ToyMLP(num_classes=NULL).eval()


def test_scale_one_is_the_conditional_model(model: ToyMLP) -> None:
    x, t, y = torch.randn(5, 2), torch.rand(5), torch.randint(0, NULL, (5,))
    guided = ClassifierFreeGuidance(model, 1.0, NULL)(x, t, y)
    assert torch.allclose(guided, model(x, t, y), atol=1e-6)


def test_scale_zero_is_the_unconditional_model(model: ToyMLP) -> None:
    x, t, y = torch.randn(5, 2), torch.rand(5), torch.randint(0, NULL, (5,))
    guided = ClassifierFreeGuidance(model, 0.0, NULL)(x, t, y)
    assert torch.allclose(guided, model(x, t, None), atol=1e-6)


def test_guidance_needs_labels(model: ToyMLP) -> None:
    with pytest.raises(ValueError):
        ClassifierFreeGuidance(model, 2.0, NULL)(torch.randn(2, 2), torch.rand(2), None)
