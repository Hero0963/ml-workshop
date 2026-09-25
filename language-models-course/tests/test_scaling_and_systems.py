# tests/test_scaling_and_systems.py
import numpy as np
import pytest
import torch

from lm_course.kernels import data_parallel_gradients, online_softmax, tiled_attention
from lm_course.model import GPT, attention_reference, gpt2_config, llama_style_config
from lm_course.quantization import (
    fake_quantize,
    quantize_absmax,
    quantize_model,
    weight_bytes,
)
from lm_course.scaling import (
    chinchilla_optimal,
    fit_isoflop_minimum,
    fit_power_law,
    kv_cache_bytes,
    matmul_parameters,
    measured_training_flops,
    saved_activation_bytes,
    training_flops_per_token,
    zero_memory_per_gpu,
)
from lm_course.training import lm_loss

CONFIGS = {
    "gpt2": gpt2_config(
        "gpt2", vocab_size=64, context_length=32, n_layer=2, n_head=4, d_model=32
    ),
    "llama_gqa": llama_style_config(
        vocab_size=64, context_length=32, n_layer=2, n_head=4, d_model=32, n_kv_head=2
    ),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_matmul_parameters_are_the_linear_weights(name: str) -> None:
    model = GPT(CONFIGS[name])
    linear = sum(
        m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear)
    )
    assert matmul_parameters(CONFIGS[name]) == linear


@pytest.mark.parametrize("name", list(CONFIGS))
def test_six_n_matches_the_flop_counter(name: str) -> None:
    config = CONFIGS[name]
    model = GPT(config)
    batch, seq = 2, config.context_length
    measured = measured_training_flops(model, torch.randint(0, 64, (batch, seq)))
    estimate = training_flops_per_token(config)
    tokens = batch * seq
    # PyTorch's fused CPU attention kernel is not seen by FlopCounterMode (torch 2.14), so the
    # count is the 6N part alone; a counter that does see it lands on the full estimate.
    assert estimate.matmul * tokens <= measured <= estimate.total * tokens * 1.01


def test_attention_term_is_twelve_d_t_per_token_and_layer() -> None:
    from torch.utils.flop_counter import FlopCounterMode

    B, H, T, D = 2, 4, 32, 8
    q, k, v = (torch.randn(B, H, T, D, requires_grad=True) for _ in range(3))
    counter = FlopCounterMode(display=False)
    with counter:
        attention_reference(q, k, v, causal=True).sum().backward()
    per_token = counter.get_total_flops() / (B * T)
    assert per_token == 12 * (H * D) * T


def test_chinchilla_split() -> None:
    n, d = chinchilla_optimal(6 * 1e9 * 20e9)
    assert n == pytest.approx(1e9)
    assert d == pytest.approx(20e9)


def test_power_law_and_isoflop_fits_recover_known_curves() -> None:
    x = np.logspace(3, 7, 10)
    a, b = fit_power_law(x, 5.0 * x**-0.3)
    assert (a, b) == (pytest.approx(5.0), pytest.approx(-0.3))
    n = np.logspace(5, 8, 9)
    losses = 0.2 * (np.log(n) - np.log(3e6)) ** 2 + 2.5
    assert fit_isoflop_minimum(n, losses) == pytest.approx(3e6, rel=1e-6)


def test_zero_stages_shard_the_model_state() -> None:
    n = 1e9
    assert zero_memory_per_gpu(n, 8, stage=0) == pytest.approx(16 * n)
    assert zero_memory_per_gpu(n, 8, stage=1) == pytest.approx(4 * n + 12 * n / 8)
    assert zero_memory_per_gpu(n, 8, stage=3) == pytest.approx(16 * n / 8)


def test_kv_cache_formula_matches_the_cache_object() -> None:
    config = CONFIGS["llama_gqa"]
    cache = GPT(config).new_kv_cache(batch_size=3, max_length=20)
    assert kv_cache_bytes(config, 3, 20, bytes_per_value=4) == cache.num_bytes()


def test_activation_memory_grows_linearly_with_batch() -> None:
    model = GPT(CONFIGS["gpt2"])
    one = saved_activation_bytes(model, torch.randint(0, 64, (1, 32)))
    four = saved_activation_bytes(model, torch.randint(0, 64, (4, 32)))
    assert four == pytest.approx(4 * one, rel=0.02)


@pytest.mark.parametrize("block", [1, 3, 16, 100])
def test_online_softmax_is_exact(block: int) -> None:
    x = torch.randn(4, 37, dtype=torch.float64) * 10
    torch.testing.assert_close(online_softmax(x, block), x.softmax(dim=-1))


@pytest.mark.parametrize("causal", [True, False])
def test_tiled_attention_is_exact(causal: bool) -> None:
    q, k, v = torch.randn(3, 2, 3, 50, 8, dtype=torch.float64).unbind(0)
    out, lse = tiled_attention(q, k, v, block_q=16, block_k=7, causal=causal)
    torch.testing.assert_close(out, attention_reference(q, k, v, causal=causal))
    scores = q @ k.transpose(-2, -1) / np.sqrt(8)
    if causal:
        scores = scores.masked_fill(torch.ones(50, 50).triu(1).bool(), float("-inf"))
    torch.testing.assert_close(lse, scores.logsumexp(dim=-1))


def test_data_parallel_average_equals_full_batch_gradient() -> None:
    model = GPT(CONFIGS["gpt2"])
    x = torch.randint(0, 64, (8, 16))
    y = torch.randint(0, 64, (8, 16))

    def loss_fn(m: torch.nn.Module, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        return lm_loss(m(xs), ys)

    loss_fn(model, x, y).backward()
    full = [p.grad.clone() for p in model.parameters()]
    parallel = data_parallel_gradients(model, loss_fn, x, y, n_workers=4)
    for a, b in zip(full, parallel):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_absmax_quantization_error_is_at_most_half_a_step() -> None:
    w = torch.randn(16, 64)
    codes, scale = quantize_absmax(w, bits=8)
    assert codes.dtype == torch.int8
    assert codes.abs().max() == 127
    assert ((codes.float() * scale - w).abs() <= scale / 2 + 1e-6).all()
    err8 = (fake_quantize(w, 8) - w).pow(2).mean()
    err4 = (fake_quantize(w, 4) - w).pow(2).mean()
    assert (
        err4 > 100 * err8
    )  # each bit removed doubles the step, i.e. 4x the squared error


def test_quantize_model_leaves_the_original_alone() -> None:
    model = GPT(CONFIGS["llama_gqa"])
    before = model.blocks[0].mlp.up_proj.weight.clone()
    q = quantize_model(model, bits=4)
    torch.testing.assert_close(model.blocks[0].mlp.up_proj.weight, before)
    assert not torch.equal(q.blocks[0].mlp.up_proj.weight, before)
    assert weight_bytes(model, 4) < weight_bytes(model, 8) < weight_bytes(model, 32)
