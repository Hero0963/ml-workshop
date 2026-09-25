# tests/test_ngram.py
import numpy as np
import pytest

from lm_course.ngram import (
    AddKLM,
    InterpolatedLM,
    bits_per_byte,
    sample_text,
    text_to_bytes,
)

TEXT = text_to_bytes("the cat sat on the mat. the dog sat on the log. " * 20)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_add_k_distributions_sum_to_one_and_match_pointwise_probs(order: int) -> None:
    model = AddKLM(order, k=0.5).fit(TEXT)
    positions = np.arange(10, 60)
    pointwise = model.probs_at(TEXT, positions)
    for pos, p in zip(positions, pointwise):
        dist = model.next_distribution(TEXT[:pos].tobytes())
        assert dist.sum() == pytest.approx(1.0)
        assert dist[TEXT[pos]] == pytest.approx(p)


def test_maximum_likelihood_bigram() -> None:
    data = text_to_bytes("abababac")
    model = AddKLM(2, k=0.0).fit(data)
    dist = model.next_distribution(b"a")
    assert dist[ord("b")] == pytest.approx(3 / 4)
    assert dist[ord("c")] == pytest.approx(1 / 4)


def test_unigram_bits_per_byte_is_the_byte_entropy() -> None:
    model = AddKLM(1, k=0.0).fit(TEXT)
    _, counts = np.unique(TEXT, return_counts=True)
    freq = counts / counts.sum()
    entropy = -(freq * np.log2(freq)).sum()
    # evaluated on (almost) the whole training text, the unigram MLE costs its entropy
    assert bits_per_byte(model, TEXT, start=0) == pytest.approx(entropy, rel=1e-6)


def test_huge_smoothing_gives_eight_bits_per_byte() -> None:
    model = AddKLM(3, k=1e9).fit(TEXT)
    assert bits_per_byte(model, TEXT) == pytest.approx(8.0, abs=1e-3)


def test_longer_contexts_fit_the_training_text_better() -> None:
    bpb = [bits_per_byte(AddKLM(n, k=0.01).fit(TEXT), TEXT) for n in (1, 2, 3, 5)]
    assert bpb == sorted(bpb, reverse=True)


def test_interpolation_sums_to_one_and_handles_unseen_contexts() -> None:
    model = InterpolatedLM(4).fit(TEXT)
    for context in (b"the ", b"zzz", b""):
        assert model.next_distribution(context).sum() == pytest.approx(1.0)
    unseen = text_to_bytes("xyzzy quux")
    assert np.all(model.probs_at(unseen, np.arange(4, len(unseen))) > 0)


def test_sampling_from_a_deterministic_model_repeats_the_text() -> None:
    model = AddKLM(6, k=0.0).fit(text_to_bytes("hello world. " * 10))
    out = sample_text(model, "hello", 20, np.random.default_rng(0))
    assert out == "hello world. hello world."
