# tests/test_training.py
import math

import numpy as np
import pytest
import torch

from lm_course.model import GPT, llama_style_config
from lm_course.optim import build_optimizer
from lm_course.tokenizer import BPETokenizer
from lm_course.training import evaluate, lm_loss, random_batch, tokenize_stories, train


class ZeroLogits(torch.nn.Module):
    """A 'model' that predicts the uniform distribution."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*idx.shape, self.vocab_size) + 0 * self.dummy


def test_random_batch_targets_are_shifted_inputs() -> None:
    tokens = np.arange(1000, dtype=np.uint16)
    x, y = random_batch(tokens, 4, 16, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 16)
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    torch.testing.assert_close(y[:, 0], x[:, 0] + 1)


def test_ignored_targets_do_not_count() -> None:
    logits = torch.randn(1, 4, 10)
    targets = torch.tensor([[1, 2, -1, -1]])
    expected = torch.nn.functional.cross_entropy(logits[0, :2], targets[0, :2])
    torch.testing.assert_close(lm_loss(logits, targets), expected)


def test_uniform_model_has_log_v_loss_and_matching_bits_per_byte() -> None:
    vocab = 64
    tokens = np.random.default_rng(0).integers(0, vocab, 2000).astype(np.uint16)
    token_bytes = torch.full((vocab,), 3)
    token_bytes[0] = 0  # a "special" token: left out of bits-per-byte
    result = evaluate(
        ZeroLogits(vocab), tokens, context_length=50, token_bytes=token_bytes
    )
    assert result.loss == pytest.approx(math.log(vocab), rel=1e-5)
    assert result.perplexity == pytest.approx(vocab, rel=1e-4)
    # every counted token costs log2(64) = 6 bits for 3 bytes
    assert result.bits_per_byte == pytest.approx(2.0, rel=1e-5)


def test_tokenize_stories_prepends_bos() -> None:
    tok = BPETokenizer.train(["one two three"] * 5, 270, special_tokens=["<|bos|>"])
    tokens = tokenize_stories(tok, ["one two", "three"])
    bos = tok.special_id("<|bos|>")
    assert tokens[0] == bos and list(tokens).count(bos) == 2
    assert tok.decode([t for t in tokens if t != bos]) == "one twothree"


def test_training_memorizes_a_repeated_sequence() -> None:
    config = llama_style_config(
        vocab_size=16, context_length=32, n_layer=2, n_head=2, d_model=32
    )
    model = GPT(config)
    pattern = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7], dtype=np.uint16)
    tokens = np.tile(pattern, 200)
    result = train(
        model,
        tokens,
        tokens,
        num_steps=150,
        batch_size=8,
        optimizer=build_optimizer(model, "adamw", lr=3e-3),
        warmup=10,
        eval_every=50,
        log_every=0,
    )
    assert result.losses[0] > 2.0
    assert result.final_eval_loss < 0.1
    assert result.tokens_seen == 150 * 8 * 32
