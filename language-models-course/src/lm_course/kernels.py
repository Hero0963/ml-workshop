# src/lm_course/kernels.py
"""What fast attention kernels compute, written in plain PyTorch (lesson 08).

``online_softmax`` and ``tiled_attention`` follow the algorithm of FlashAttention (Dao et al.
2022; Milakov & Gimelshein 2018 for the online normalizer): process keys block by block and
keep a running maximum and running sum, so the T x T score matrix never exists in full.
In Python they are slower than the fused kernel; the point is that they are exact.
"""

import math
import time
from collections.abc import Callable

import torch
from torch import nn


def online_softmax(x: torch.Tensor, block: int) -> torch.Tensor:
    """softmax over the last dimension, reading ``x`` one block at a time.

    Invariant after each block: m = max so far, s = sum of exp(x - m) so far. When a larger
    maximum shows up, the old sum is rescaled by exp(m_old - m_new).
    """
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype)
    s = torch.zeros(x.shape[:-1], dtype=x.dtype)
    for start in range(0, x.shape[-1], block):
        chunk = x[..., start : start + block]
        m_new = torch.maximum(m, chunk.max(dim=-1).values)
        s = s * torch.exp(m - m_new) + torch.exp(chunk - m_new[..., None]).sum(dim=-1)
        m = m_new
    return torch.exp(x - m[..., None]) / s[..., None]


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int = 64,
    block_k: int = 64,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention's forward pass: exact attention output and the log-sum-exp per query.

    q, k, v: (B, H, T, D). For each block of queries, loop over blocks of keys, keeping
    m (running max of scores), l (running sum of exp) and acc (running weighted sum of values).
    The largest intermediate is a block_q x block_k tile, not T x T.
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    lse = torch.empty(B, H, T, dtype=q.dtype)
    for qs in range(0, T, block_q):
        qe = min(qs + block_q, T)
        q_blk = q[:, :, qs:qe] * scale
        m = torch.full((B, H, qe - qs), float("-inf"), dtype=q.dtype)
        l = torch.zeros(B, H, qe - qs, dtype=q.dtype)  # noqa: E741 - the paper's name
        acc = torch.zeros(B, H, qe - qs, D, dtype=q.dtype)
        k_end = (
            qe if causal else T
        )  # keys after the last query of this block are all masked
        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            scores = q_blk @ k[:, :, ks:ke].transpose(-2, -1)  # (B, H, bq, bk)
            if causal and ke > qs:
                rows = torch.arange(qs, qe)[:, None]
                cols = torch.arange(ks, ke)[None, :]
                scores = scores.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, scores.max(dim=-1).values)
            p = torch.exp(scores - m_new[..., None])
            correction = torch.exp(m - m_new)
            l = l * correction + p.sum(dim=-1)  # noqa: E741
            acc = acc * correction[..., None] + p @ v[:, :, ks:ke]
            m = m_new
        out[:, :, qs:qe] = acc / l[..., None]
        lse[:, :, qs:qe] = m + torch.log(l)
    return out, lse


def attention_score_bytes(
    seq_len: int, n_head: int, batch_size: int, bytes_per_value: int = 2
) -> int:
    """Memory of the full score (or probability) matrix that naive attention materializes."""
    return batch_size * n_head * seq_len * seq_len * bytes_per_value


def time_it(fn: Callable[[], object], repeats: int = 5, warmup: int = 2) -> float:
    """Median wall-clock seconds of ``fn()``."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def matmul_throughput(
    n: int, dtype: torch.dtype = torch.float32, repeats: int = 5
) -> dict[str, float]:
    """Achieved FLOP/s and arithmetic intensity of an n x n x n matrix multiply.

    Intensity = FLOPs / bytes moved = 2 n^3 / (3 n^2 * bytes): it grows with n, which is why
    big matmuls are compute-bound and small ones memory-bound (the roofline picture).
    """
    a, b = torch.randn(n, n, dtype=dtype), torch.randn(n, n, dtype=dtype)
    seconds = time_it(lambda: a @ b, repeats)
    flops = 2 * n**3
    moved = 3 * n * n * a.element_size()
    return {
        "n": n,
        "gflops": flops / seconds / 1e9,
        "intensity": flops / moved,
        "seconds": seconds,
    }


def elementwise_bandwidth(numel: int, repeats: int = 5) -> dict[str, float]:
    """Achieved memory bandwidth of y = a + b (1 FLOP per 12 bytes moved in fp32)."""
    a, b = torch.randn(numel), torch.randn(numel)
    seconds = time_it(lambda: a + b, repeats)
    moved = 3 * numel * a.element_size()
    return {
        "numel": numel,
        "gb_per_s": moved / seconds / 1e9,
        "intensity": numel / moved,
    }


def data_parallel_gradients(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    n_workers: int,
) -> list[torch.Tensor]:
    """Simulate data parallelism: split the batch across ``n_workers`` replicas, compute each
    replica's gradient, then all-reduce (average). With a mean loss and equal shards the result
    equals the full-batch gradient, which is why DDP needs no change to the math."""
    grads = None
    for xs, ys in zip(x.chunk(n_workers), y.chunk(n_workers)):
        model.zero_grad(set_to_none=True)
        loss_fn(model, xs, ys).backward()
        local = [p.grad.detach().clone() for p in model.parameters()]
        grads = local if grads is None else [g + lg for g, lg in zip(grads, local)]
    model.zero_grad(set_to_none=True)
    return [g / n_workers for g in grads]
