# src/lm_course/training.py
"""Pretraining loop for the GPT (lessons 06 and 07): batches, loss, evaluation, logging.

Kept deliberately small: one process, one device, gradient accumulation, clipping,
a learning-rate schedule, periodic validation. Lesson 08 explains what changes at scale.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from lm_course.model import GPT
from lm_course.optim import CombinedOptimizer, build_optimizer, cosine_schedule
from lm_course.tokenizer import BPETokenizer

IGNORE_INDEX = -1


def tokenize_stories(
    tokenizer: BPETokenizer, stories: list[str], cache_path: Path | None = None
) -> np.ndarray:
    """One flat stream of token ids; every story is preceded by the document separator
    ``<|bos|>`` so the model learns where documents start (as in nanochat)."""
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    bos = tokenizer.special_id("<|bos|>")
    ids: list[int] = []
    for story in stories:
        ids.append(bos)
        ids.extend(tokenizer.encode_ordinary(story))
    dtype = (
        np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max + 1 else np.int32
    )
    tokens = np.array(ids, dtype=dtype)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, tokens)
    return tokens


def random_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    generator: np.random.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of ``context_length + 1`` tokens; targets are inputs shifted by one."""
    starts = generator.integers(0, len(tokens) - context_length - 1, size=batch_size)
    windows = np.stack([tokens[s : s + context_length + 1] for s in starts]).astype(
        np.int64
    )
    batch = torch.from_numpy(windows).to(device)
    return batch[:, :-1], batch[:, 1:]


def lm_loss(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Cross-entropy of the next token, in nats; targets equal to -1 are ignored."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )
    return loss if reduction != "none" else loss.view(targets.shape)


@dataclass
class EvalResult:
    loss: float  # nats per token
    bits_per_byte: float | None = None

    @property
    def perplexity(self) -> float:
        return math.exp(self.loss)


@torch.no_grad()
def evaluate(
    model: GPT,
    tokens: np.ndarray,
    context_length: int,
    batch_size: int = 32,
    max_batches: int = 20,
    token_bytes: torch.Tensor | None = None,
) -> EvalResult:
    """Loss on consecutive, non-overlapping windows from the start of ``tokens`` (the same
    windows every call, so numbers are comparable across a run).

    With ``token_bytes`` (bytes per token id, 0 for special tokens) it also returns
    bits-per-byte: total nats / (ln 2 * total bytes), which does not depend on the vocabulary.
    """
    model.eval()
    device = next(model.parameters()).device
    n_windows = min(max_batches * batch_size, (len(tokens) - 1) // context_length)
    total_nats, total_tokens = 0.0, 0
    byte_nats, total_bytes = 0.0, 0
    for start in range(0, n_windows, batch_size):
        rows = range(start, min(start + batch_size, n_windows))
        windows = np.stack(
            [
                tokens[r * context_length : r * context_length + context_length + 1]
                for r in rows
            ]
        ).astype(np.int64)
        batch = torch.from_numpy(windows).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        nats = lm_loss(model(x), y, reduction="none")
        total_nats += nats.sum().item()
        total_tokens += y.numel()
        if token_bytes is not None:
            nbytes = token_bytes.to(device)[y]
            # special tokens (0 bytes) are left out of the numerator as well
            byte_nats += (nats * (nbytes > 0)).sum().item()
            total_bytes += int(nbytes.sum().item())
    model.train()
    bpb = byte_nats / (math.log(2) * total_bytes) if total_bytes else None
    return EvalResult(total_nats / total_tokens, bpb)


@dataclass
class TrainResult:
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    tokens_seen: int = 0

    @property
    def final_eval_loss(self) -> float:
        return self.eval_losses[-1]


def train(
    model: GPT,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray | None = None,
    *,
    num_steps: int,
    batch_size: int = 32,
    context_length: int | None = None,
    grad_accum: int = 1,
    optimizer: CombinedOptimizer | None = None,
    lr: float = 3e-3,
    schedule: Callable[[int, int, int], float] = cosine_schedule,
    warmup: int = 100,
    grad_clip: float | None = 1.0,
    eval_every: int = 250,
    eval_batches: int = 10,
    log_every: int = 100,
    seed: int = 0,
) -> TrainResult:
    """Train ``model`` for ``num_steps`` optimizer steps of ``batch_size * grad_accum`` windows."""
    device = next(model.parameters()).device
    context_length = context_length or model.config.context_length
    optimizer = optimizer or build_optimizer(model, "adamw", lr=lr)
    rng = np.random.default_rng(seed)
    result = TrainResult()
    model.train()
    start = time.perf_counter()
    for step in range(num_steps):
        optimizer.set_lr_multiplier(schedule(step, num_steps, warmup))
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = random_batch(train_tokens, batch_size, context_length, rng, device)
            loss = lm_loss(model(x), y) / grad_accum
            loss.backward()
            step_loss += loss.item()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        result.steps.append(step)
        result.losses.append(step_loss)
        result.tokens_seen += batch_size * grad_accum * context_length
        last = step == num_steps - 1
        if val_tokens is not None and ((step + 1) % eval_every == 0 or last):
            val = evaluate(model, val_tokens, context_length, batch_size, eval_batches)
            result.eval_steps.append(step + 1)
            result.eval_losses.append(val.loss)
        if log_every and ((step + 1) % log_every == 0 or last):
            val_text = (
                f", val {result.eval_losses[-1]:.3f}" if result.eval_losses else ""
            )
            logger.info(
                f"step {step + 1:5d}/{num_steps}: loss {step_loss:.3f}{val_text}"
                f" ({time.perf_counter() - start:.0f}s)"
            )
    result.seconds = time.perf_counter() - start
    model.eval()
    return result
