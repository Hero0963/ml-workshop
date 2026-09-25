# src/lm_course/ngram.py
"""Count-based byte-level n-gram language models (lesson 01).

The counts are stored as sorted integer keys: a context of n-1 bytes is read as a base-256
number, so "context followed by byte b" is the key ``context * 256 + b``. Looking up a
probability is a binary search, and fitting on 20 MB of text takes a few seconds with numpy.
"""

import math

import numpy as np

VOCAB = 256
MAX_ORDER = 7  # 256 ** 7 still fits in an int64 key


def text_to_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def _context_keys(data: np.ndarray, order: int, positions: np.ndarray) -> np.ndarray:
    """Base-256 number formed by the ``order - 1`` bytes before each position."""
    keys = np.zeros(len(positions), dtype=np.int64)
    for offset in range(order - 1, 0, -1):
        keys = keys * VOCAB + data[positions - offset].astype(np.int64)
    return keys


class NGramCounts:
    """Counts of (context, next byte) and of contexts, for one order n."""

    def __init__(self, order: int) -> None:
        if not 1 <= order <= MAX_ORDER:
            raise ValueError(f"order must be in 1..{MAX_ORDER}")
        self.order = order
        self.pair_keys = np.zeros(0, dtype=np.int64)
        self.pair_counts = np.zeros(0, dtype=np.int64)
        self.context_keys = np.zeros(0, dtype=np.int64)
        self.context_counts = np.zeros(0, dtype=np.int64)

    def fit(self, data: np.ndarray) -> "NGramCounts":
        positions = np.arange(self.order - 1, len(data))
        contexts = _context_keys(data, self.order, positions)
        pairs = contexts * VOCAB + data[positions].astype(np.int64)
        self.pair_keys, self.pair_counts = np.unique(pairs, return_counts=True)
        self.context_keys, self.context_counts = np.unique(contexts, return_counts=True)
        return self

    @staticmethod
    def _lookup(keys: np.ndarray, counts: np.ndarray, query: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(keys, query)
        idx = np.minimum(idx, len(keys) - 1)
        return np.where(keys[idx] == query, counts[idx], 0)

    def counts_at(
        self, data: np.ndarray, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """c(context, next) and c(context) for the byte at each position of ``data``."""
        contexts = _context_keys(data, self.order, positions)
        pairs = contexts * VOCAB + data[positions].astype(np.int64)
        return (
            self._lookup(self.pair_keys, self.pair_counts, pairs),
            self._lookup(self.context_keys, self.context_counts, contexts),
        )

    def next_counts(self, context: bytes) -> np.ndarray:
        """c(context, b) for all 256 bytes b after one given context (zeros if it is too short)."""
        if len(context) < self.order - 1:
            return np.zeros(VOCAB, dtype=np.int64)
        ctx = np.frombuffer(context[len(context) - (self.order - 1) :], dtype=np.uint8)
        key = 0
        for byte in ctx:
            key = key * VOCAB + int(byte)
        query = key * VOCAB + np.arange(VOCAB, dtype=np.int64)
        return self._lookup(self.pair_keys, self.pair_counts, query)

    @property
    def num_distinct_contexts(self) -> int:
        return len(self.context_keys)


class AddKLM:
    """P(b | context) = (c(context, b) + k) / (c(context) + 256 k). With k = 0 this is the
    maximum-likelihood estimate, which gives probability 0 to anything unseen."""

    def __init__(self, order: int, k: float = 1.0) -> None:
        self.order = order
        self.k = k
        self.counts = NGramCounts(order)

    def fit(self, data: np.ndarray) -> "AddKLM":
        self.counts.fit(data)
        return self

    def probs_at(self, data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        pair, context = self.counts.counts_at(data, positions)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (pair + self.k) / (context + self.k * VOCAB)

    def next_distribution(self, context: bytes) -> np.ndarray:
        counts = self.counts.next_counts(context).astype(np.float64) + self.k
        total = counts.sum()
        return counts / total if total > 0 else np.full(VOCAB, 1.0 / VOCAB)


class InterpolatedLM:
    """Mix the maximum-likelihood estimates of every order 1..n (Jelinek-Mercer):

    P(b | ctx) = sum_j w_j P_ML_j(b | last j-1 bytes of ctx) + w_0 / 256,

    where an order whose context was never seen drops out and the remaining weights are
    renormalized. Short contexts are always available, long ones are used when they help.
    """

    def __init__(self, order: int, weights: list[float] | None = None) -> None:
        self.order = order
        # default: geometric weights that favor longer contexts; weights[0] is the uniform
        self.weights = weights or [0.5**i for i in range(order, -1, -1)]
        if len(self.weights) != order + 1:
            raise ValueError(
                "need one weight per order plus one for the uniform distribution"
            )
        self.levels = [NGramCounts(j) for j in range(1, order + 1)]

    def fit(self, data: np.ndarray) -> "InterpolatedLM":
        for level in self.levels:
            level.fit(data)
        return self

    def probs_at(self, data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        mixed = np.full(len(positions), self.weights[0] / VOCAB)
        total_weight = np.full(len(positions), self.weights[0])
        for weight, level in zip(self.weights[1:], self.levels):
            pair, context = level.counts_at(data, positions)
            seen = context > 0
            mixed += np.where(seen, weight * pair / np.maximum(context, 1), 0.0)
            total_weight += np.where(seen, weight, 0.0)
        return mixed / total_weight

    def next_distribution(self, context: bytes) -> np.ndarray:
        mixed = np.full(VOCAB, self.weights[0] / VOCAB)
        total_weight = self.weights[0]
        for weight, level in zip(self.weights[1:], self.levels):
            counts = level.next_counts(context)
            if counts.sum() > 0:
                mixed += weight * counts / counts.sum()
                total_weight += weight
        return mixed / total_weight


def bits_per_byte(
    model: AddKLM | InterpolatedLM, data: np.ndarray, start: int = 8
) -> float:
    """Average -log2 P(byte | context) over positions ``start`` onwards (the same positions for
    every order, so the numbers are comparable)."""
    positions = np.arange(start, len(data))
    probs = model.probs_at(data, positions)
    with np.errstate(divide="ignore"):
        return float(-np.mean(np.log2(probs)))


def sample_text(
    model: AddKLM | InterpolatedLM,
    prompt: str,
    num_bytes: int,
    rng: np.random.Generator,
) -> str:
    out = bytearray(prompt.encode("utf-8"))
    for _ in range(num_bytes):
        probs = model.next_distribution(bytes(out))
        out.append(int(rng.choice(VOCAB, p=probs / probs.sum())))
    return out.decode("utf-8", errors="replace")


def perplexity_from_bits(bits_per_unit: float) -> float:
    return 2.0**bits_per_unit


def nats_to_bits(nats: float) -> float:
    return nats / math.log(2)
