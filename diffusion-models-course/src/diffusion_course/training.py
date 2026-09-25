# src/diffusion_course/training.py
"""One training loop for every lab: any model, any loss, any stream of batches."""

import copy
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from torch import nn

from diffusion_course.models import build_model

LossFn = Callable[[nn.Module, torch.Tensor, torch.Tensor | None], torch.Tensor]

DEFAULT_LR = 2e-4
DEFAULT_EMA_DECAY = 0.999
GRAD_CLIP_NORM = 1.0


class EMA:
    """Exponential moving average of the weights.

    Diffusion samples come out noticeably cleaner from the averaged weights than from the
    last optimiser step, whose weights still jitter with the minibatch noise.
    """

    def __init__(self, model: nn.Module, decay: float = DEFAULT_EMA_DECAY) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    ema: EMA | None = None
    seconds: float = 0.0

    @property
    def sampling_model(self) -> nn.Module | None:
        return self.ema.model if self.ema is not None else None


def train(
    model: nn.Module,
    loss_fn: LossFn,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor | None]],
    num_steps: int,
    lr: float = DEFAULT_LR,
    ema_decay: float | None = DEFAULT_EMA_DECAY,
    device: torch.device | str = "cpu",
    log_every: int = 500,
) -> TrainResult:
    """Plain AdamW with gradient clipping; returns per-step losses and the EMA model."""
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    result = TrainResult(ema=EMA(model, ema_decay) if ema_decay else None)
    start = time.perf_counter()
    for step in range(1, num_steps + 1):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device) if y is not None else None
        loss = loss_fn(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        if result.ema is not None:
            result.ema.update(model)
        result.losses.append(loss.item())
        if log_every and step % log_every == 0:
            recent = sum(result.losses[-log_every:]) / log_every
            logger.info(f"step {step}/{num_steps}  loss {recent:.4f}")
    result.seconds = time.perf_counter() - start
    model.eval()
    return result


def save_checkpoint(
    path: Path, model: nn.Module, arch: str, model_kwargs: dict, **metadata: object
) -> None:
    """Store the weights together with everything needed to rebuild the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            **metadata,
        },
        path,
    )
    logger.info(f"saved checkpoint to {path}")


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint; returns (model in eval mode, full checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["arch"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
