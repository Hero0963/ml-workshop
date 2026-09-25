# src/lm_course/sampling.py
"""Turning next-token distributions into text (lessons 05 and 09).

``generate`` is the plain autoregressive loop, with or without a KV cache.
``speculative_generate`` lets a small draft model propose tokens that the large model verifies,
and still samples exactly from the large model's distribution.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from lm_course.model import GPT


def filter_logits(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Set to -inf every logit outside the top-k and outside the nucleus of mass top_p."""
    logits = logits.clone()
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits[logits < kth] = float("-inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        # drop a token when the tokens ranked above it already cover top_p
        drop_sorted = probs.cumsum(dim=-1) - probs >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        logits[drop] = float("-inf")
    return logits


def next_token_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """The distribution actually sampled from, after temperature, top-k and top-p."""
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        return probs.scatter(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return filter_logits(logits / temperature, top_k, top_p).softmax(dim=-1)


def sample_from(
    probs: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: GPT,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    num_samples: int = 1,
    stop_tokens: set[int] | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Sample ``num_samples`` continuations of one prompt, as a batch.

    With the cache, the prompt is processed once ("prefill") and every later step feeds only
    the newest token ("decode"). Without it, every step re-runs the whole sequence; the outputs
    are identical, only slower (lesson 09 §2.1). Generation stops early once every sample has
    produced a stop token; the stop token is kept in the output.
    """
    model.eval()
    device = next(model.parameters()).device
    context = model.config.context_length
    max_new_tokens = min(max_new_tokens, context - len(prompt))
    tokens = torch.tensor([prompt] * num_samples, device=device)
    cache = (
        model.new_kv_cache(num_samples, len(prompt) + max_new_tokens)
        if use_cache
        else None
    )
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)
    outputs: list[list[int]] = [[] for _ in range(num_samples)]
    step_input = tokens
    for _ in range(max_new_tokens):
        logits = model(step_input if use_cache else tokens, kv_cache=cache)[:, -1]
        next_ids = sample_from(
            next_token_probs(logits, temperature, top_k, top_p), generator
        )
        for i in range(num_samples):
            if not finished[i]:
                outputs[i].append(int(next_ids[i]))
                if stop_tokens and int(next_ids[i]) in stop_tokens:
                    finished[i] = True
        if finished.all():
            break
        tokens = torch.cat([tokens, next_ids[:, None]], dim=1)
        step_input = next_ids[:, None]
    return outputs


# ---------------------------------------------------------------------------------------------
# Speculative decoding (Leviathan et al. 2022; Chen et al. 2023)
# ---------------------------------------------------------------------------------------------


@dataclass
class SpeculativeStats:
    proposed: int = 0  # drafted tokens
    checked: int = (
        0  # drafted tokens the target looked at (the rest follow a rejection)
    )
    accepted: int = 0
    target_calls: int = 0
    draft_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.checked, 1)


def speculative_step(
    target_probs_fn: Callable[[list[int]], torch.Tensor],
    draft_probs_fn: Callable[[list[int]], torch.Tensor],
    prefix: list[int],
    num_draft: int,
    generator: torch.Generator | None = None,
    stats: SpeculativeStats | None = None,
) -> list[int]:
    """One round: the draft proposes ``num_draft`` tokens, the target scores all of them in a
    single call, and each proposal x is kept with probability min(1, p(x) / q(x)).

    On the first rejection a replacement is drawn from the residual max(0, p - q), normalized;
    if everything is accepted, one bonus token comes from p at the end. Either way the returned
    tokens are distributed exactly as if sampled from the target one by one.

    ``target_probs_fn(seq)`` returns the target's next-token distributions for the last
    ``num_draft + 1`` positions of ``seq`` (shape (num_draft + 1, V)); ``draft_probs_fn(seq)``
    returns the draft's distribution after ``seq`` (shape (V,)).
    """
    stats = stats or SpeculativeStats()
    proposal, draft_probs = [], []
    seq = list(prefix)
    for _ in range(num_draft):
        q = draft_probs_fn(seq)
        x = int(sample_from(q, generator))
        proposal.append(x)
        draft_probs.append(q)
        seq.append(x)
    stats.draft_calls += num_draft
    target_probs = target_probs_fn(seq)  # rows: after prefix, after prefix + x1, ...
    stats.target_calls += 1
    stats.proposed += num_draft

    accepted: list[int] = []
    for i, x in enumerate(proposal):
        p, q = target_probs[i], draft_probs[i]
        stats.checked += 1
        u = torch.rand((), generator=generator)
        if u < torch.clamp(p[x] / q[x], max=1.0):
            accepted.append(x)
            stats.accepted += 1
            continue
        residual = torch.clamp(p - q, min=0.0)
        residual = residual / residual.sum()
        accepted.append(int(sample_from(residual, generator)))
        return accepted
    accepted.append(int(sample_from(target_probs[num_draft], generator)))
    return accepted


@torch.no_grad()
def speculative_generate(
    target: GPT,
    draft: GPT,
    prompt: list[int],
    max_new_tokens: int,
    num_draft: int = 4,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[list[int], SpeculativeStats]:
    """Speculative sampling with two GPTs sharing a tokenizer. Kept cache-free for clarity:
    it shows the algorithm and counts target calls, not the wall-clock gain (lesson 09 §2.4)."""
    target.eval()
    draft.eval()
    device = next(target.parameters()).device

    def target_probs(seq: list[int]) -> torch.Tensor:
        logits = target(torch.tensor([seq], device=device))[0, -(num_draft + 1) :]
        return next_token_probs(logits, temperature).cpu()

    def draft_probs(seq: list[int]) -> torch.Tensor:
        logits = draft(torch.tensor([seq], device=device))[0, -1]
        return next_token_probs(logits, temperature).cpu()

    stats = SpeculativeStats()
    seq = list(prompt)
    while len(seq) - len(prompt) < max_new_tokens:
        seq += speculative_step(
            target_probs, draft_probs, seq, num_draft, generator, stats
        )
    return seq[len(prompt) : len(prompt) + max_new_tokens], stats
