# tests/test_models.py
import pytest
import torch

from diffusion_course.models import TinyDiT, ToyMLP, UNet, build_model


@pytest.mark.parametrize("num_classes", [None, 10])
def test_toy_mlp_keeps_the_shape(num_classes: int | None) -> None:
    y = torch.randint(0, 10, (7,)) if num_classes else None
    out = ToyMLP(num_classes=num_classes)(torch.randn(7, 2), torch.rand(7), y)
    assert out.shape == (7, 2)


@pytest.mark.parametrize("arch", [UNet, TinyDiT])
@pytest.mark.parametrize("num_classes", [None, 10])
def test_image_models_start_at_zero_and_learn(
    arch: type, num_classes: int | None
) -> None:
    model = arch(num_classes=num_classes)
    x = torch.randn(3, 1, 28, 28)
    y = torch.randint(0, 10, (3,)) if num_classes else None
    out = model(x, torch.rand(3), y)
    assert out.shape == x.shape
    assert torch.all(out == 0)  # zero-initialised output layer
    ((out - x) ** 2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_unconditional_call_uses_the_null_class() -> None:
    model = UNet(num_classes=10)
    x, t = torch.randn(2, 1, 28, 28), torch.rand(2)
    null = torch.full((2,), 10)
    assert torch.equal(model(x, t, None), model(x, t, null))


def test_dit_unpatchify_inverts_patchify_layout() -> None:
    model = TinyDiT(image_size=8, patch_size=4, in_channels=2)
    image = torch.randn(3, 2, 8, 8)
    b, c, g, p = 3, 2, 2, 4
    tokens = (
        image.reshape(b, c, g, p, g, p)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(b, g * g, p * p * c)
    )
    assert torch.equal(model.unpatchify(tokens), image)


def test_build_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError):
        build_model("vae")
