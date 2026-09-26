# tests/test_word2vec.py
import numpy as np
import pytest
import torch

from lm_course.data import Analogy, toy_world_analogies, toy_world_corpus
from lm_course.word2vec import (
    SkipGram,
    analogy_accuracy,
    build_vocab,
    cooccurrence_matrix,
    encode_sentences,
    keep_probabilities,
    noise_distribution,
    pmi_matrix,
    sgns_objective,
    shifted_ppmi,
    skipgram_pairs,
    solve_analogy,
    svd_embeddings,
    words_from_text,
)


def test_vocab_is_sorted_by_frequency_and_drops_rare_words() -> None:
    vocab = build_vocab([["a", "b", "a"], ["c", "a", "b"]], min_count=2)
    assert vocab.words == ["a", "b"]
    assert vocab.counts.tolist() == [3, 2]
    assert encode_sentences([["a", "c", "b"]], vocab)[0].tolist() == [0, 1]


def test_window_one_pairs_are_the_neighbours_in_both_directions() -> None:
    centers, contexts = skipgram_pairs(
        [np.array([0, 1, 2])], 1, np.random.default_rng(0)
    )
    assert sorted(zip(centers.tolist(), contexts.tolist())) == [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    ]


def test_subsampling_and_noise_formulas() -> None:
    counts = np.array([9000, 900, 90, 10])
    keep = keep_probabilities(counts, t=1e-2)
    assert keep[0] == pytest.approx(np.sqrt(1e-2 / 0.9))
    assert keep[-1] == 1.0
    noise = noise_distribution(counts)
    assert noise.sum() == pytest.approx(1.0)
    assert noise[0] < counts[0] / counts.sum()  # frequent words are down-weighted
    assert noise[-1] > counts[-1] / counts.sum()


def test_sgns_loss_matches_the_formula() -> None:
    model = SkipGram(5, 3)
    torch.nn.init.normal_(model.context.weight)
    c, o, neg = torch.tensor([1]), torch.tensor([2]), torch.tensor([[3, 4]])
    v, u = model.center.weight, model.context.weight
    expected = -(
        torch.log(torch.sigmoid(u[2] @ v[1]))
        + torch.log(torch.sigmoid(-u[3] @ v[1]))
        + torch.log(torch.sigmoid(-u[4] @ v[1]))
    )
    torch.testing.assert_close(model.loss(c, o, neg), expected)


def test_sgns_optimum_is_shifted_pmi() -> None:
    """Levy & Goldberg (2014): with enough dimensions, the SGNS dot products converge to
    PMI(w, c) - log k (here with the plain unigram as noise, so P_n(c) = #c / |D|)."""
    rng = np.random.default_rng(0)
    cooc = torch.from_numpy(rng.integers(1, 50, size=(6, 6)).astype(np.float64)).float()
    noise = cooc.sum(dim=0) / cooc.sum()
    k = 3
    model = SkipGram(6, 6)
    torch.nn.init.normal_(model.center.weight, std=0.1)
    torch.nn.init.normal_(model.context.weight, std=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(3000):
        opt.zero_grad()
        dots = model.center.weight @ model.context.weight.T
        sgns_objective(dots, cooc, noise, k).backward()
        opt.step()
    dots = (model.center.weight @ model.context.weight.T).detach().numpy()
    expected = pmi_matrix(cooc.double().numpy()) - np.log(k)
    np.testing.assert_allclose(dots, expected, atol=0.05)


def test_cooccurrence_counts() -> None:
    cooc = cooccurrence_matrix([np.array([0, 1, 2, 1])], 3, window=1)
    assert cooc.tolist() == [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
    cooc2 = cooccurrence_matrix([np.array([0, 1, 2, 1])], 3, window=2)
    np.testing.assert_array_equal(cooc2, cooc2.T)
    assert cooc2[0, 2] == 1


def test_shifted_ppmi_is_non_negative_and_svd_factorizes() -> None:
    cooc = np.random.default_rng(0).integers(0, 20, size=(8, 8)).astype(float)
    cooc = cooc + cooc.T
    m = shifted_ppmi(cooc, k=2)
    assert (m >= 0).all()
    psd = m @ m.T
    w = svd_embeddings(psd, 8)
    np.testing.assert_allclose(w @ w.T, psd, atol=1e-8)


def test_analogy_by_vector_offset() -> None:
    from lm_course.word2vec import Vocab

    vocab = Vocab(["a", "b", "c", "d", "e"], np.ones(5, dtype=np.int64))
    emb = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]], dtype=float
    )  # b - a = d - c
    assert solve_analogy(emb, vocab, "a", "b", "c") == "d"
    assert (
        analogy_accuracy(emb, vocab, [Analogy("a", "b", "c", "d", "x")])["all"] == 1.0
    )


def test_toy_world_is_deterministic_and_covers_every_analogy_word() -> None:
    corpus = toy_world_corpus(20_000, seed=3)
    assert corpus == toy_world_corpus(20_000, seed=3)
    words = {w for sentence in corpus for w in sentence}
    for q in toy_world_analogies():
        assert {q.a, q.b, q.c, q.d} <= words


def test_toy_world_roles_are_what_tell_son_from_boy() -> None:
    def contexts(corpus: list[list[str]], word: str) -> set[str]:
        return {w for sentence in corpus if word in sentence for w in sentence} - {word}

    with_roles = toy_world_corpus(20_000, seed=0)
    without = toy_world_corpus(20_000, seed=0, with_roles=False)
    assert contexts(with_roles, "son") != contexts(with_roles, "boy")
    assert contexts(without, "son") == contexts(without, "boy")


def test_words_from_text() -> None:
    assert words_from_text("Lily's dog ran! It was 3 o'clock.") == [
        "lily's",
        "dog",
        "ran",
        "it",
        "was",
        "o'clock",
    ]
