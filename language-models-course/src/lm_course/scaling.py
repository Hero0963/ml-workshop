# src/lm_course/scaling.py
"""Resource accounting and scaling laws (lesson 07).

FLOPs: every weight in a matrix multiply costs 2 FLOPs per token forward (a multiply and an
add) and 4 backward, hence the classic 6 N D. Attention adds a term that grows with context.
Memory: parameters, gradients and optimizer state are counted per parameter; activations are
measured by hooking the tensors autograd saves.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lm_course.model import GPTConfig

BYTES_FP32 = 4
BYTES_BF16 = 2
CHINCHILLA_TOKENS_PER_PARAM = 20


def matmul_parameters(config: GPTConfig) -> int:
    """Weights that multiply the token stream: attention and MLP matrices plus the LM head.
    Embedding lookups cost no FLOPs, so they are excluded (as in nanochat and PaLM)."""
    d, hd = config.d_model, config.head_dim
    attention = (
        d * config.n_head * hd * 2 + d * config.kv_heads * hd * 2
    )  # q, out; k, v
    mlp_matrices = 3 if config.mlp == "swiglu" else 2
    mlp = mlp_matrices * d * config.ff_dim
    return config.n_layer * (attention + mlp) + d * config.vocab_size


@dataclass
class FlopEstimate:
    matmul: float  # 6 * matmul parameters, per token
    attention: (
        float  # 12 * n_layer * d * T, per token (QK^T and AV, forward + backward)
    )

    @property
    def total(self) -> float:
        return self.matmul + self.attention


def training_flops_per_token(
    config: GPTConfig, seq_len: int | None = None
) -> FlopEstimate:
    """PaLM-style estimate (Chowdhery et al. 2022, appendix B): 6N + 12 L d T per token.

    The attention term counts the full T x T score matrix; a causal kernel that skips the
    masked half does about half of it.
    """
    seq_len = seq_len or config.context_length
    attention = 12 * config.n_layer * config.n_head * config.head_dim * seq_len
    return FlopEstimate(6 * matmul_parameters(config), attention)


def measured_training_flops(model: nn.Module, idx: torch.Tensor) -> int:
    """FLOPs of one forward + backward pass, counted by PyTorch's FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter:
        model(idx).float().pow(2).mean().backward()
    model.zero_grad(set_to_none=True)
    return counter.get_total_flops()


def chinchilla_optimal(
    compute: float, tokens_per_param: float = CHINCHILLA_TOKENS_PER_PARAM
) -> tuple[float, float]:
    """Model size N and tokens D with C = 6 N D and D = r N (Hoffmann et al. 2022: r ~ 20)."""
    n = math.sqrt(compute / (6 * tokens_per_param))
    return n, tokens_per_param * n


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y = a * x^b fitted as a straight line in log-log space; returns (a, b)."""
    b, log_a = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_a)), float(b)


def fit_isoflop_minimum(params: np.ndarray, losses: np.ndarray) -> float:
    """Fit loss = a (log N)^2 + b log N + c along one IsoFLOP curve; return the N at the
    bottom of the parabola (the compute-optimal size for that budget, Chinchilla approach 2)."""
    a, b, _ = np.polyfit(np.log(params), losses, 2)
    return float(np.exp(-b / (2 * a)))


# ---------------------------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------------------------

# bytes per parameter for weights + gradients + optimizer state (Rajbhandari et al. 2019, §3.1)
BYTES_PER_PARAM = {
    "fp32 SGD": 4 + 4,
    "fp32 AdamW": 4 + 4 + 8,
    "mixed-precision AdamW": 2
    + 2
    + (4 + 8),  # bf16 weights and grads, fp32 master + moments
}


def zero_memory_per_gpu(
    n_params: float,
    n_gpus: int,
    stage: int,
    weight_bytes: int = 2,
    optimizer_bytes: int = 12,
) -> float:
    """Bytes per GPU for model state under ZeRO data parallelism.

    Stage 0 replicates everything; stage 1 shards the optimizer state; stage 2 also the
    gradients; stage 3 also the weights.
    """
    weights, grads, optim = (
        weight_bytes * n_params,
        weight_bytes * n_params,
        optimizer_bytes * n_params,
    )
    if stage >= 1:
        optim /= n_gpus
    if stage >= 2:
        grads /= n_gpus
    if stage >= 3:
        weights /= n_gpus
    return weights + grads + optim


def kv_cache_bytes(
    config: GPTConfig, batch_size: int, seq_len: int, bytes_per_value: int = 2
) -> int:
    """Keys and values of every layer: 2 * L * n_kv_head * head_dim * T * B values."""
    per_token = 2 * config.n_layer * config.kv_heads * config.head_dim
    return per_token * seq_len * batch_size * bytes_per_value


def saved_activation_bytes(model: nn.Module, idx: torch.Tensor) -> int:
    """Bytes of the tensors autograd keeps for the backward pass of one forward.

    Parameters are excluded (they live in memory anyway); tensors saved several times are
    counted once.
    """
    param_ptrs = {p.data_ptr() for p in model.parameters()}
    seen: set[tuple[int, int]] = set()
    total = 0

    def pack(t: torch.Tensor) -> torch.Tensor:
        nonlocal total
        key = (t.data_ptr(), t.numel())
        if t.data_ptr() not in param_ptrs and key not in seen:
            seen.add(key)
            total += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(idx)
    return total
