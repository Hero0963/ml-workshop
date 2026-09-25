# src/lm_course/optim.py
"""Optimizers and learning-rate schedules (lesson 06).

- AdamW for everything (the GPT-2 / GPT-3 / CS336 recipe), or
- Muon for the hidden 2D matrices and AdamW for embeddings, the LM head and the vectors
  (the modded-nanogpt / nanochat recipe).
"""

import math

import torch
from torch import nn

# Coefficients of the quintic Newton-Schulz iteration from Keller Jordan's Muon post (2024-12)
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


def newton_schulz_orthogonalize(g: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """Push every singular value of ``g`` towards 1 while keeping its singular vectors.

    For G = U S V^T this approximates U V^T, the orthogonal matrix closest to G. Each step
    applies the odd polynomial p(s) = a s + b s^3 + c s^5 to the singular values; the
    coefficients are tuned for speed, so the result has singular values roughly in [0.7, 1.2]
    rather than exactly 1 (lesson 06 §2.5).
    """
    a, b, c = NS_COEFFICIENTS
    x = g / (g.norm() + 1e-7)  # Frobenius norm >= spectral norm, so all s <= 1
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz, for 2D weight matrices only.

    update = orthogonalize(nesterov_momentum(grad)) * sqrt(max(1, rows / cols)).
    The shape factor keeps the per-element update size comparable across rectangular matrices.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001 - torch.optim signature
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError(
                        "Muon only handles 2D parameters; use AdamW for the rest"
                    )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = (
                    p.grad.add(buf, alpha=group["momentum"])
                    if group["nesterov"]
                    else buf
                )
                update = newton_schulz_orthogonalize(g)
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)


class CombinedOptimizer:
    """Several optimizers stepped together, with one shared learning-rate multiplier."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        for opt in optimizers:
            for group in opt.param_groups:
                group.setdefault("initial_lr", group["lr"])

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def set_lr_multiplier(self, multiplier: float) -> None:
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * multiplier

    def state_dict(self) -> list[dict]:
        return [opt.state_dict() for opt in self.optimizers]


def split_parameters(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Hidden matrices (inside the blocks), embeddings / LM head, and 1D vectors."""
    groups: dict[str, list[nn.Parameter]] = {
        "matrices": [],
        "embeddings": [],
        "vectors": [],
    }
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.dim() < 2:
            groups["vectors"].append(p)
        elif name.startswith("blocks."):
            groups["matrices"].append(p)
        else:
            groups["embeddings"].append(p)
    return groups


def build_optimizer(
    model: nn.Module,
    kind: str = "adamw",
    lr: float = 3e-3,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    muon_lr: float = 0.02,
) -> CombinedOptimizer:
    """``kind="adamw"``: AdamW with weight decay on matrices only (GPT-3 style).
    ``kind="muon"``: Muon on the hidden matrices, AdamW (same ``lr``) on everything else."""
    groups = split_parameters(model)
    no_decay = {"params": groups["vectors"], "weight_decay": 0.0}
    if kind == "adamw":
        decay = {
            "params": groups["matrices"] + groups["embeddings"],
            "weight_decay": weight_decay,
        }
        return CombinedOptimizer(
            [torch.optim.AdamW([decay, no_decay], lr=lr, betas=betas)]
        )
    if kind == "muon":
        adam = torch.optim.AdamW(
            [{"params": groups["embeddings"], "weight_decay": weight_decay}, no_decay],
            lr=lr,
            betas=betas,
        )
        return CombinedOptimizer([Muon(groups["matrices"], lr=muon_lr), adam])
    raise ValueError(f"unknown optimizer {kind!r}")


# ---------------------------------------------------------------------------------------------
# Learning-rate schedules: multipliers in [0, 1] applied to each group's initial lr
# ---------------------------------------------------------------------------------------------


def cosine_schedule(
    step: int, total: int, warmup: int, min_ratio: float = 0.1
) -> float:
    """Linear warmup, then cosine decay to ``min_ratio`` (GPT-3, CS336 assignment 1)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def wsd_schedule(
    step: int, total: int, warmup: int, decay_fraction: float = 0.4
) -> float:
    """Warmup-stable-decay: linear warmup, constant, then linear decay to 0 over the last
    ``decay_fraction`` of training (nanochat's "warmdown")."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = total - int(decay_fraction * total)
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1, total - decay_start))
