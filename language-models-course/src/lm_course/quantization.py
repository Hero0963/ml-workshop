# src/lm_course/quantization.py
"""Weight-only post-training quantization (lesson 09).

Symmetric "absmax" quantization per output row: each row of a weight matrix gets its own
scale s = max|w| / (2^(bits-1) - 1), and w is stored as round(w / s), an integer in
[-(2^(bits-1) - 1), 2^(bits-1) - 1]. Real kernels multiply with the integers; here the weights
are quantized and immediately dequantized ("fake quantization") to measure the damage.
"""

import copy

import torch
from torch import nn


def quantize_absmax(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer codes (as int8 storage for bits <= 8) and one scale per row."""
    q_max = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / q_max
    codes = torch.round(w / scale).clamp(-q_max, q_max)
    return codes.to(torch.int8 if bits <= 8 else torch.int32), scale


def dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.float() * scale


def fake_quantize(w: torch.Tensor, bits: int) -> torch.Tensor:
    return dequantize(*quantize_absmax(w, bits)).to(w.dtype)


def quantize_model(model: nn.Module, bits: int, include_head: bool = True) -> nn.Module:
    """A copy of ``model`` whose Linear weights went through ``bits``-bit quantization.
    Embedding tables, norms and biases stay in full precision (common practice), except that
    with tied embeddings the LM head *is* the embedding table and gets quantized with it."""
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in quantized.named_modules():
            if isinstance(module, nn.Linear) and (include_head or name != "lm_head"):
                module.weight.copy_(fake_quantize(module.weight, bits))
    return quantized


def weight_bytes(model: nn.Module, bits: int) -> int:
    """Storage for Linear weights at ``bits`` per value plus one fp16 scale per row;
    everything else counted at fp32."""
    total = 0
    seen: set[int] = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            if isinstance(module, nn.Linear) and name == "weight":
                total += p.numel() * bits // 8 + p.shape[0] * 2
            else:
                total += p.numel() * 4
    return total
