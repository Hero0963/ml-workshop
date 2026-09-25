# src/diffusion_course/models/__init__.py
"""Networks used in the course. All share the call signature model(x, t, y=None)."""

from torch import nn

from diffusion_course.models.dit import TinyDiT
from diffusion_course.models.mlp import ToyMLP
from diffusion_course.models.unet import UNet

ARCHITECTURES: dict[str, type[nn.Module]] = {
    "mlp": ToyMLP,
    "unet": UNet,
    "dit": TinyDiT,
}


def build_model(arch: str, **kwargs: object) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {arch!r}; choose from {list(ARCHITECTURES)}"
        )
    return ARCHITECTURES[arch](**kwargs)


__all__ = ["ARCHITECTURES", "TinyDiT", "ToyMLP", "UNet", "build_model"]
