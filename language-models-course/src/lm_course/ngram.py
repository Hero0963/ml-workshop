# src/lm_course/ngram.py
"""Count-based byte-level n-gram language models (lesson 01).

The counts are stored as sorted integer keys: a context of n-1 bytes is hashed to a 55-bit
integer, so "context followed by byte b" is the key ``hash(context) * 256 + b``. Looking up a
probability is a binary search, and fitting on 20 MB of text takes a few seconds with numpy.
(Two different contexts could share a hash, but with ~10 million contexts and 2^55 possible
hashes the chance that any pair collides is about 0.1%, and one collision would barely matter.)
"""

import math

import numpy as np

VOCAB = 256
HASH_BITS = 55  # hash * 256 + byte must fit in an int64
HASH_MULTIPLIER = np.uint64(
    0x9E3779B97F4A7C15
)  # odd, so each step is a bijection mod 2^64


def text_to_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def _context_keys(data: np.ndarray, order: int, positions: np.ndarray) -> np.ndarray:
    """A hash of the ``order - 1`` bytes before each position (0 for the empty context)."""
    keys = np.zeros(len(positions), dtype=np.uint64)
    for offset in range(order - 1, 0, -1):
        byte = data[positions - offset].astype(np.uint64) + np.uint64(1)
        keys = (
            keys + byte
        ) * HASH_MULTIPLIER  # uint64 arithmetic wraps around mod 2^64
    # the top bits of a product depend on all the bits of its inputs
    return (keys >> np.uint64(64 - HASH_BITS)).astype(np.int64)


class NGramCounts:
    """Counts of (context, next byte) and of contexts, for one order n, plus the number of
    distinct bytes seen after each context (what Witten-Bell smoothing needs)."""

    def __init__(self, order: int) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.pair_keys = np.zeros(0, dtype=np.int64)
        self.pair_counts = np.zeros(0, dtype=np.int64)
        self.context_keys = np.zeros(0, dtype=np.int64)
        self.context_counts = np.zeros(0, dtype=np.int64)
        self.context_types = np.zeros(0, dtype=np.int64)

    def fit(self, data: np.ndarray) -> "NGramCounts":
        positions = np.arange(self.order - 1, len(data))
        contexts = _context_keys(data, self.order, positions)
        pairs = contexts * VOCAB + data[positions].astype(np.int64)
        self.pair_keys, self.pair_counts = np.unique(pairs, return_counts=True)
        self.context_keys, self.context_counts = np.unique(contexts, return_counts=True)
        # pair keys are sorted by context first, so this lines up with context_keys
        _, self.context_types = np.unique(self.pair_keys // VOCAB, return_counts=True)
        return self

    @staticmethod
    def _lookup(keys: np.ndarray, counts: np.ndarray, query: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(keys, query)
        idx = np.minimum(idx, len(keys) - 1)
        return np.where(keys[idx] == query, counts[idx], 0)

    def counts_at(
        self, data: np.ndarray, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """c(context, next), c(context) and the number of distinct bytes seen after the
        context, for the byte at each position of ``data``."""
        contexts = _context_keys(data, self.order, positions)
        pairs = contexts * VOCAB + data[positions].astype(np.int64)
        return (
            self._lookup(self.pair_keys, self.pair_counts, pairs),
            self._lookup(self.context_keys, self.context_counts, contexts),
            self._lookup(self.context_keys, self.context_types, contexts),
        )

    def next_counts(self, context: bytes) -> np.ndarray:
        """c(context, b) for all 256 bytes b after one given context (zeros if it is too short)."""
        if len(context) < self.order - 1:
            return np.zeros(VOCAB, dtype=np.int64)
        data = np.frombuffer(context + b"\0", dtype=np.uint8)
        key = _context_keys(data, self.order, np.array([len(context)]))[0]
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

    @classmethod
    def from_counts(cls, counts: NGramCounts, k: float = 1.0) -> "AddKLM":
        """Reuse counts that were already fitted (e.g. one level of an InterpolatedLM)."""
        model = cls(counts.order, k)
        model.counts = counts
        return model

    def probs_at(self, data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        pair, context, _ = self.counts.counts_at(data, positions)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (pair + self.k) / (context + self.k * VOCAB)

    def next_distribution(self, context: bytes) -> np.ndarray:
        counts = self.counts.next_counts(context).astype(np.float64) + self.k
        total = counts.sum()
        return counts / total if total > 0 else np.full(VOCAB, 1.0 / VOCAB)


class InterpolatedLM:
    """Interpolate every order from the uniform distribution up to n (Witten-Bell):

    P_j(b | h) = lam(h) P_ML(b | h) + (1 - lam(h)) P_{j-1}(b | h shortened by one byte),
    lam(h) = c(h) / (c(h) + u(h)),  P_0(b) = 1 / 256,

    where u(h) is the number of distinct bytes seen after h. A context seen often, with few
    different continuations, is trusted; a rare or unseen one (lam = 0) hands its probability
    to the shorter context. u(h) is an estimate of how likely a new continuation is.
    """

    def __init__(self, order: int) -> None:
        self.order = order
        self.levels = [NGramCounts(j) for j in range(1, order + 1)]

    def fit(self, data: np.ndarray) -> "InterpolatedLM":
        for level in self.levels:
            level.fit(data)
        return self

    def up_to(self, order: int) -> "InterpolatedLM":
        """The same model using only orders 1..order (shares the fitted counts)."""
        model = InterpolatedLM(0)
        model.order, model.levels = order, self.levels[:order]
        return model

    def probs_at(self, data: np.ndarray, positions: np.ndarray) -> np.ndarray:
        probs = np.full(len(positions), 1.0 / VOCAB)
        for level in self.levels:
            pair, context, types = level.counts_at(data, positions)
            lam = context / np.maximum(context + types, 1)
            probs = lam * pair / np.maximum(context, 1) + (1 - lam) * probs
        return probs

    def next_distribution(self, context: bytes) -> np.ndarray:
        probs = np.full(VOCAB, 1.0 / VOCAB)
        for level in self.levels:
            counts = level.next_counts(context)
            total, types = counts.sum(), np.count_nonzero(counts)
            if total > 0:
                lam = total / (total + types)
                probs = lam * counts / total + (1 - lam) * probs
        return probs


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
