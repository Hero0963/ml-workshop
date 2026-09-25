# tests/test_optim.py
import pytest
import torch

from lm_course.model import GPT, gpt2_config, llama_style_config
from lm_course.optim import (
    Muon,
    build_optimizer,
    cosine_schedule,
    newton_schulz_orthogonalize,
    split_parameters,
    wsd_schedule,
)


@pytest.mark.parametrize("shape", [(64, 64), (32, 96), (96, 32)])
def test_newton_schulz_flattens_the_spectrum_and_keeps_the_directions(
    shape: tuple[int, int],
) -> None:
    # a badly conditioned matrix, like a momentum-averaged gradient
    u, _ = torch.linalg.qr(torch.randn(shape[0], shape[0]))
    v, _ = torch.linalg.qr(torch.randn(shape[1], shape[1]))
    k = min(shape)
    s = torch.logspace(0, -3, k)
    g = u[:, :k] @ torch.diag(s) @ v[:, :k].T
    x = newton_schulz_orthogonalize(g)
    singular = torch.linalg.svdvals(x)
    assert singular.max() < 1.3
    assert singular[: k // 2].min() > 0.6  # the large directions are all pushed near 1
    polar = u[:, :k] @ v[:, :k].T  # the exact answer U V^T
    cosine = (x * polar).sum() / (x.norm() * polar.norm())
    assert cosine > 0.9


def test_muon_decreases_a_least_squares_loss() -> None:
    w = torch.nn.Parameter(torch.zeros(16, 8))
    target = torch.randn(16, 8)
    opt = Muon([w], lr=0.05)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = (w - target).pow(2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.05 * losses[0]


def test_muon_rejects_vectors() -> None:
    b = torch.nn.Parameter(torch.zeros(4))
    b.grad = torch.ones(4)
    with pytest.raises(ValueError):
        Muon([b]).step()


def test_parameter_split_covers_every_tensor_once() -> None:
    model = GPT(
        gpt2_config(
            "gpt2", vocab_size=50, context_length=16, n_layer=2, n_head=2, d_model=32
        )
    )
    groups = split_parameters(model)
    ids = [id(p) for ps in groups.values() for p in ps]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert all(p.dim() == 2 for p in groups["matrices"])
    # tied: the LM head is the token embedding, so the embedding group has wte and wpe only
    assert len(groups["embeddings"]) == 2


@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_build_optimizer_steps_every_parameter(kind: str) -> None:
    model = GPT(
        llama_style_config(
            vocab_size=50, context_length=16, n_layer=2, n_head=2, d_model=32
        )
    )
    before = [p.detach().clone() for p in model.parameters()]
    opt = build_optimizer(model, kind, lr=1e-2)
    logits = model(torch.randint(0, 50, (2, 8)))
    logits.square().mean().backward()
    opt.step()
    changed = [not torch.equal(a, b) for a, b in zip(before, model.parameters())]
    assert all(changed)


def test_schedules() -> None:
    assert cosine_schedule(0, 1000, 100) == pytest.approx(0.01)
    assert cosine_schedule(99, 1000, 100) == pytest.approx(1.0)
    assert cosine_schedule(999, 1000, 100) == pytest.approx(0.1, abs=1e-4)
    assert wsd_schedule(500, 1000, 100, decay_fraction=0.4) == 1.0
    assert wsd_schedule(800, 1000, 100, decay_fraction=0.4) == pytest.approx(0.5)
    assert wsd_schedule(1000, 1000, 100, decay_fraction=0.4) == 0.0
