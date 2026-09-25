# tests/test_training.py
from pathlib import Path

import torch

from diffusion_course.data import toy_batches
from diffusion_course.ddpm import ddpm_loss
from diffusion_course.models import ToyMLP
from diffusion_course.schedules import NoiseSchedule
from diffusion_course.training import load_checkpoint, save_checkpoint, train


def test_training_reduces_the_loss_and_tracks_an_ema() -> None:
    schedule = NoiseSchedule.create("linear", 100)
    model = ToyMLP(hidden_dim=64)
    result = train(
        model,
        lambda m, x, y: ddpm_loss(m, schedule, x, y),
        toy_batches("moons", 256),
        num_steps=300,
        lr=1e-3,
        log_every=0,
    )
    first, last = result.losses[:50], result.losses[-50:]
    assert sum(last) / 50 < sum(first) / 50
    ema_weight = result.sampling_model.input.weight
    assert not torch.equal(ema_weight, model.input.weight)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    kwargs = {"hidden_dim": 32, "num_blocks": 2}
    model = ToyMLP(**kwargs).eval()
    path = tmp_path / "toy.pt"
    save_checkpoint(path, model, "mlp", kwargs, method="ddpm")
    restored, checkpoint = load_checkpoint(path)
    x, t = torch.randn(4, 2), torch.rand(4)
    assert checkpoint["method"] == "ddpm"
    assert torch.equal(restored(x, t), model(x, t))
