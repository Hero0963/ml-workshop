# tests/test_embeddings.py
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lm_course.embeddings import (
    TextEncoder,
    alignment_and_uniformity,
    average_word_vectors,
    bm25_scores,
    info_nce_loss,
    last_token_pool,
    matryoshka_loss,
    mean_pool,
    random_crops,
    retrieval_metrics,
    split_halves,
    tokenize_batch,
)
from lm_course.model import GPT, llama_style_config
from lm_course.tokenizer import BPETokenizer


def test_pooling_ignores_padding() -> None:
    hidden = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    torch.testing.assert_close(mean_pool(hidden, mask)[0], hidden[0, :2].mean(dim=0))
    torch.testing.assert_close(
        last_token_pool(hidden, mask), torch.stack([hidden[0, 1], hidden[1, 2]])
    )


def test_info_nce_is_log_batch_size_for_random_and_small_for_matched() -> None:
    q = F.normalize(torch.randn(256, 64), dim=-1)
    random_docs = F.normalize(torch.randn(256, 64), dim=-1)
    assert info_nce_loss(q, random_docs, temperature=1.0).item() == pytest.approx(
        np.log(256), rel=0.02
    )
    assert info_nce_loss(q, q, temperature=0.05).item() < 0.01


def test_matryoshka_loss_averages_prefix_losses() -> None:
    q, d = torch.randn(8, 16), torch.randn(8, 16)
    expected = (
        info_nce_loss(F.normalize(q[:, :4], dim=-1), F.normalize(d[:, :4], dim=-1))
        + info_nce_loss(F.normalize(q, dim=-1), F.normalize(d, dim=-1))
    ) / 2
    torch.testing.assert_close(matryoshka_loss(q, d, [4, 16]), expected)


def test_retrieval_metrics() -> None:
    scores = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.5, 0.1], [0.0, 0.2, 0.1]])
    metrics = retrieval_metrics(scores, ks=(1, 2))
    assert metrics["recall@1"] == pytest.approx(
        1 / 3
    )  # only query 0 ranks its doc first
    assert metrics["recall@2"] == 1.0  # query 2 ranks its doc second
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 2) / 3)


def test_bm25_prefers_rare_matching_words() -> None:
    docs = ["the cat sat on the mat", "the dog ate the bone", "the the the the"]
    scores = bm25_scores(["cat", "dog bone", "the"], docs)
    assert scores[0].argmax() == 0 and scores[1].argmax() == 1
    assert scores[0, 0] > scores[2].max()  # a rare word is worth more than "the"


def test_alignment_uniformity_extremes() -> None:
    x = F.normalize(torch.randn(100, 8), dim=-1)
    align, uniform = alignment_and_uniformity(x, x)
    assert align == 0.0
    collapsed = F.normalize(torch.ones(100, 8), dim=-1)
    assert alignment_and_uniformity(collapsed, collapsed)[1] == pytest.approx(
        0.0, abs=1e-6
    )
    assert uniform < -2.0  # spread-out vectors


def test_views_of_a_text() -> None:
    text = "One. Two two. Three three three. Four."
    assert split_halves(text) == ("One. Two two.", "Three three three. Four.")
    a, b = random_crops("a b c d e f g h i j", random.Random(0))
    assert a and b and set(a.split()) <= set("abcdefghij")


def test_average_word_vectors() -> None:
    vectors = np.eye(3)
    out = average_word_vectors(["a b", "c zzz"], vectors, {"a": 0, "b": 1, "c": 2})
    torch.testing.assert_close(
        out[0], F.normalize(torch.tensor([1.0, 1.0, 0.0]), dim=0)
    )
    torch.testing.assert_close(out[1], torch.tensor([0.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    "pooling, causal", [("mean", True), ("last", True), ("mean", False)]
)
def test_encoder_outputs_unit_vectors_independent_of_padding(
    pooling: str, causal: bool
) -> None:
    tok = BPETokenizer.train(
        ["a small dog ran to the big tree"] * 5, 300, special_tokens=["<|bos|>"]
    )
    config = llama_style_config(
        vocab_size=tok.vocab_size,
        context_length=32,
        n_layer=1,
        n_head=2,
        d_model=16,
        causal=causal,
    )
    encoder = TextEncoder(GPT(config), pooling=pooling).eval()
    short = encoder.encode(tok, ["a small dog"], max_tokens=32)
    padded = encoder.encode(
        tok, ["a small dog", "a small dog ran to the big tree"], max_tokens=32
    )
    torch.testing.assert_close(short[0], padded[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(padded.norm(dim=-1), torch.ones(2))
    ids, mask = tokenize_batch(tok, ["a", "a small dog"], 32)
    assert (
        ids[:, 0].eq(tok.special_id("<|bos|>")).all()
        and mask.sum(dim=1).tolist()[0] == 2
    )
