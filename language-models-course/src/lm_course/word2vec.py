# src/lm_course/word2vec.py
"""word2vec (lesson 02): skip-gram with negative sampling, and the count-based view (PPMI + SVD).

Follows Mikolov et al. (2013a, 2013b) for the model, the subsampling of frequent words and the
unigram^0.75 noise distribution, and Levy & Goldberg (2014) for the matrix-factorization view.
Written from the papers; no code from the original C implementation.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

SUBSAMPLE_T = 1e-4
NOISE_POWER = 0.75


@dataclass
class Vocab:
    words: list[str]
    counts: np.ndarray
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index


def build_vocab(sentences: list[list[str]], min_count: int = 5) -> Vocab:
    """Words seen at least ``min_count`` times, most frequent first."""
    counter = Counter(w for sentence in sentences for w in sentence)
    kept = [(w, c) for w, c in counter.most_common() if c >= min_count]
    return Vocab([w for w, _ in kept], np.array([c for _, c in kept], dtype=np.int64))


def encode_sentences(sentences: list[list[str]], vocab: Vocab) -> list[np.ndarray]:
    """Word ids per sentence; out-of-vocabulary words are dropped."""
    return [
        np.array([vocab.index[w] for w in sentence if w in vocab.index], dtype=np.int64)
        for sentence in sentences
    ]


def keep_probabilities(counts: np.ndarray, t: float = SUBSAMPLE_T) -> np.ndarray:
    """Mikolov et al. (2013b): discard each occurrence of word w with probability
    1 - sqrt(t / f(w)), f = relative frequency. Rare words are always kept."""
    freq = counts / counts.sum()
    return np.minimum(1.0, np.sqrt(t / freq))


def noise_distribution(counts: np.ndarray, power: float = NOISE_POWER) -> np.ndarray:
    """P_n(w) proportional to count(w)^0.75: flattens the unigram distribution a little."""
    weights = counts.astype(np.float64) ** power
    return weights / weights.sum()


def skipgram_pairs(
    sentences: list[np.ndarray],
    window: int,
    rng: np.random.Generator,
    keep_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(center, context) pairs. For every center word a window size b is drawn from 1..window,
    so nearby words are paired more often than distant ones (as in the original tool)."""
    centers, contexts = [], []
    for ids in sentences:
        if keep_probs is not None:
            ids = ids[rng.random(len(ids)) < keep_probs[ids]]
        n = len(ids)
        if n < 2:
            continue
        spans = rng.integers(1, window + 1, size=n)
        for offset in range(1, window + 1):
            left = np.arange(offset, n)  # center at i, context at i - offset
            use = spans[left] >= offset
            centers.append(ids[left[use]])
            contexts.append(ids[left[use] - offset])
            right = np.arange(0, n - offset)  # center at i, context at i + offset
            use = spans[right] >= offset
            centers.append(ids[right[use]])
            contexts.append(ids[right[use] + offset])
    return np.concatenate(centers), np.concatenate(contexts)


class SkipGram(nn.Module):
    """Two tables: ``center`` vectors v_w (the embeddings we keep) and ``context`` vectors u_c.
    Initialized like the original: small uniform centers, zero contexts."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def loss(
        self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        """Negative sampling: -log s(u_o . v_c) - sum_k log s(-u_k . v_c), averaged over pairs.

        centers, contexts: (B,); negatives: (B, K) word ids drawn from the noise distribution.
        """
        v = self.center(centers)  # (B, d)
        positive = (self.context(contexts) * v).sum(-1)  # (B,)
        negative = torch.einsum("bkd,bd->bk", self.context(negatives), v)  # (B, K)
        return -(F.logsigmoid(positive) + F.logsigmoid(-negative).sum(-1)).mean()

    def embeddings(self) -> np.ndarray:
        return self.center.weight.detach().cpu().numpy()


@dataclass
class Word2VecResult:
    model: SkipGram
    losses: list[float]
    seconds: float
    num_pairs: int


def train_skipgram(
    sentences: list[np.ndarray],
    vocab: Vocab,
    dim: int = 50,
    window: int = 4,
    negatives: int = 5,
    epochs: int = 3,
    batch_size: int = 4096,
    lr: float = 0.01,
    subsample_t: float | None = SUBSAMPLE_T,
    seed: int = 0,
    log_every: int = 500,
) -> Word2VecResult:
    """Skip-gram with negative sampling, trained with Adam on shuffled minibatches of pairs.
    (The original tool uses plain SGD one pair at a time; Adam on batches is simpler to run
    here and reaches the same kind of solution.)"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = SkipGram(len(vocab), dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise = torch.from_numpy(noise_distribution(vocab.counts)).float()
    keep = keep_probabilities(vocab.counts, subsample_t) if subsample_t else None
    losses: list[float] = []
    num_pairs = 0
    start = time.perf_counter()
    step = 0
    for epoch in range(epochs):
        centers, contexts = skipgram_pairs(sentences, window, rng, keep)
        order = rng.permutation(len(centers))
        centers, contexts = (
            torch.from_numpy(centers[order]),
            torch.from_numpy(contexts[order]),
        )
        num_pairs += len(centers)
        for i in range(0, len(centers), batch_size):
            c, o = centers[i : i + batch_size], contexts[i : i + batch_size]
            neg = torch.multinomial(noise, len(c) * negatives, replacement=True).view(
                -1, negatives
            )
            loss = model.loss(c, o, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if log_every and step % log_every == 0:
                logger.info(
                    f"epoch {epoch} step {step}: loss {np.mean(losses[-log_every:]):.3f}"
                )
    return Word2VecResult(model, losses, time.perf_counter() - start, num_pairs)


# ---------------------------------------------------------------------------------------------
# The count-based view: SGNS implicitly factorizes a shifted PMI matrix (Levy & Goldberg 2014)
# ---------------------------------------------------------------------------------------------


def cooccurrence_matrix(
    sentences: list[np.ndarray], vocab_size: int, window: int
) -> np.ndarray:
    """#(w, c): how often c appears within ``window`` words of w (both directions)."""
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for ids in sentences:
        for offset in range(1, window + 1):
            if len(ids) > offset:
                np.add.at(matrix, (ids[offset:], ids[:-offset]), 1.0)
                np.add.at(matrix, (ids[:-offset], ids[offset:]), 1.0)
    return matrix


def pmi_matrix(cooc: np.ndarray, context_power: float = 1.0) -> np.ndarray:
    """PMI(w, c) = log( #(w,c) |D| / (#w #c) ), -inf where #(w,c) = 0.

    ``context_power`` = 0.75 smooths the context distribution the way the SGNS noise does."""
    total = cooc.sum()
    word = cooc.sum(axis=1, keepdims=True) / total
    context = cooc.sum(axis=0, keepdims=True) ** context_power
    context = context / context.sum()
    with np.errstate(divide="ignore"):
        return np.log(cooc / total) - np.log(word) - np.log(context)


def shifted_ppmi(
    cooc: np.ndarray, k: int = 1, context_power: float = 1.0
) -> np.ndarray:
    """max(PMI - log k, 0): the matrix SGNS with k negatives approximates, clipped at 0."""
    return np.maximum(pmi_matrix(cooc, context_power) - math.log(k), 0.0)


def svd_embeddings(matrix: np.ndarray, dim: int) -> np.ndarray:
    """W = U_d sqrt(S_d): split the singular values evenly between words and contexts."""
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :dim] * np.sqrt(s[:dim])


def sgns_objective(
    dots: torch.Tensor, cooc: torch.Tensor, noise: torch.Tensor, k: int
) -> torch.Tensor:
    """The SGNS loss summed over the whole corpus in closed form, as a function of all the
    dot products x_wc = v_w . u_c (Levy & Goldberg 2014, eq. 2):

    -sum_{w,c} #(w,c) log s(x_wc) - k sum_w #w sum_c P_n(c) log s(-x_wc).

    Its minimum is at x_wc = log( #(w,c) / (k #w P_n(c)) ), a shifted PMI.
    """
    word_counts = cooc.sum(dim=1, keepdim=True)
    positive = cooc * F.logsigmoid(dots)
    negative = k * word_counts * noise[None, :] * F.logsigmoid(-dots)
    return -(positive + negative).sum() / cooc.sum()


# ---------------------------------------------------------------------------------------------
# Using the vectors
# ---------------------------------------------------------------------------------------------


def normalize_rows(emb: np.ndarray) -> np.ndarray:
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)


def most_similar(
    emb: np.ndarray, vocab: Vocab, word: str, topn: int = 5
) -> list[tuple[str, float]]:
    unit = normalize_rows(emb)
    scores = unit @ unit[vocab.index[word]]
    order = [i for i in np.argsort(-scores) if vocab.words[i] != word][:topn]
    return [(vocab.words[i], float(scores[i])) for i in order]


def solve_analogy(emb: np.ndarray, vocab: Vocab, a: str, b: str, c: str) -> str:
    """a : b :: c : ?  ->  the word closest (cosine) to b - a + c, excluding a, b, c (3CosAdd)."""
    unit = normalize_rows(emb)
    target = unit[vocab.index[b]] - unit[vocab.index[a]] + unit[vocab.index[c]]
    scores = unit @ target
    for i in (vocab.index[a], vocab.index[b], vocab.index[c]):
        scores[i] = -np.inf
    return vocab.words[int(np.argmax(scores))]


def analogy_accuracy(
    emb: np.ndarray, vocab: Vocab, analogies: list
) -> dict[str, float]:
    """Accuracy per relation (and "all") over analogies whose four words are in the vocabulary.
    Each analogy is tried in both directions (a:b::c:d and c:d::a:b)."""
    hits: dict[str, list[bool]] = {}
    for q in analogies:
        if not all(w in vocab for w in (q.a, q.b, q.c, q.d)):
            continue
        for a, b, c, d in ((q.a, q.b, q.c, q.d), (q.c, q.d, q.a, q.b)):
            ok = solve_analogy(emb, vocab, a, b, c) == d
            hits.setdefault(q.relation, []).append(ok)
            hits.setdefault("all", []).append(ok)
    return {relation: float(np.mean(v)) for relation, v in hits.items()}


def words_from_text(text: str) -> list[str]:
    """Lower-case words and apostrophes, e.g. "Lily's dog ran!" -> ["lily's", "dog", "ran"]."""
    cleaned = "".join(ch.lower() if ch.isalpha() or ch == "'" else " " for ch in text)
    return cleaned.split()
