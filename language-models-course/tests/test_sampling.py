# tests/test_sampling.py
import pytest
import torch

from lm_course.model import GPT, llama_style_config
from lm_course.sampling import (
    SpeculativeStats,
    filter_logits,
    generate,
    next_token_probs,
    speculative_step,
)


def test_top_k_keeps_exactly_k() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    kept = torch.isfinite(filter_logits(logits, top_k=2))
    assert kept.tolist() == [[False, True, False, True, False]]


def test_top_p_keeps_the_smallest_set_reaching_p() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    kept = torch.isfinite(filter_logits(probs.log(), top_p=0.75)).tolist()
    assert kept == [[True, True, False, False]]  # 0.5 < 0.75 <= 0.5 + 0.3
    kept = torch.isfinite(filter_logits(probs.log(), top_p=0.5)).tolist()
    assert kept == [[True, False, False, False]]


def test_temperature_zero_is_argmax_and_low_temperature_sharpens() -> None:
    logits = torch.tensor([[0.0, 1.0, 0.5]])
    assert next_token_probs(logits, temperature=0).tolist() == [[0.0, 1.0, 0.0]]
    hot, cold = next_token_probs(logits, 2.0), next_token_probs(logits, 0.5)
    assert cold[0, 1] > next_token_probs(logits)[0, 1] > hot[0, 1]


def test_generation_with_and_without_cache_is_identical() -> None:
    config = llama_style_config(
        vocab_size=50, context_length=64, n_layer=2, n_head=2, d_model=32
    )
    model = GPT(config).eval()
    kwargs = dict(max_new_tokens=20, temperature=1.0, num_samples=3)
    a = generate(model, [1, 2, 3], **kwargs, generator=torch.Generator().manual_seed(7))
    b = generate(
        model,
        [1, 2, 3],
        **kwargs,
        use_cache=False,
        generator=torch.Generator().manual_seed(7),
    )
    assert a == b
    assert all(len(sample) == 20 for sample in a)


def test_generation_stops_at_stop_token() -> None:
    config = llama_style_config(
        vocab_size=5, context_length=64, n_layer=1, n_head=2, d_model=16
    )
    model = GPT(config).eval()
    outputs = generate(model, [0], 50, stop_tokens={4}, num_samples=8)
    for sample in outputs:
        assert 4 not in sample[:-1]
        assert sample[-1] == 4 or len(sample) == 50


def test_speculative_sampling_reproduces_the_target_distribution() -> None:
    p = torch.tensor([0.1, 0.4, 0.2, 0.25, 0.05])  # target
    q = torch.tensor([0.3, 0.1, 0.3, 0.1, 0.2])  # a poor draft
    num_draft = 3

    def target_probs(seq: list[int]) -> torch.Tensor:
        return p.expand(num_draft + 1, -1)

    def draft_probs(seq: list[int]) -> torch.Tensor:
        return q

    gen = torch.Generator().manual_seed(0)
    stats = SpeculativeStats()
    trials = 20_000
    first, second = torch.zeros(5), torch.zeros(5)
    for _ in range(trials):
        out = speculative_step(target_probs, draft_probs, [], num_draft, gen, stats)
        first[out[0]] += 1
        if len(out) > 1:
            second[out[1]] += 1
    torch.testing.assert_close(first / trials, p, atol=0.012, rtol=0)
    torch.testing.assert_close(second / second.sum(), p, atol=0.015, rtol=0)
    # expected acceptance of each checked proposal = sum_x min(p(x), q(x))
    assert stats.acceptance_rate == pytest.approx(
        torch.minimum(p, q).sum().item(), abs=0.01
    )
