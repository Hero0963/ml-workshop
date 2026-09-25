# src/lm_course/data_pipeline.py
"""From raw web-like text to training data (lesson 10): quality rules, deduplication,
model-based filtering, contamination checks.

The rules follow the Gopher paper (Rae et al. 2021, appendix A.1.1 and Table A1); MinHash is
Broder (1997); LSH banding is the standard construction (Leskovec, Rajaraman & Ullman, ch. 3).
"""

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")
MERSENNE_PRIME = (1 << 31) - 1
BULLETS = ("•", "-", "*", "·", "▪")


# ---------------------------------------------------------------------------------------------
# Gopher quality rules
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityRules:
    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0
    max_symbol_to_word: float = 0.1  # for "#" and for "..."
    max_bullet_lines: float = 0.9
    max_ellipsis_lines: float = 0.3
    min_alpha_words: float = 0.8
    min_stop_words: int = 2
    max_duplicate_lines: float = 0.3
    max_top_2gram_chars: float = 0.2


def gopher_failures(text: str, rules: QualityRules = QualityRules()) -> list[str]:
    """Names of the rules ``text`` breaks (an empty list means the document is kept)."""
    words = text.split()
    failures = []
    if not rules.min_words <= len(words) <= rules.max_words:
        failures.append("word count")
    if not words:
        return failures
    mean_length = sum(len(w) for w in words) / len(words)
    if not rules.min_mean_word_length <= mean_length <= rules.max_mean_word_length:
        failures.append("mean word length")
    if (
        max(text.count("#"), text.count("...") + text.count("…")) / len(words)
        > rules.max_symbol_to_word
    ):
        failures.append("symbol-to-word ratio")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        if (
            sum(line.startswith(BULLETS) for line in lines) / len(lines)
            > rules.max_bullet_lines
        ):
            failures.append("bullet lines")
        if (
            sum(line.endswith(("...", "…")) for line in lines) / len(lines)
            > rules.max_ellipsis_lines
        ):
            failures.append("ellipsis lines")
        if 1 - len(set(lines)) / len(lines) > rules.max_duplicate_lines:
            failures.append("duplicate lines")
    if (
        sum(any(ch.isalpha() for ch in w) for w in words) / len(words)
        < rules.min_alpha_words
    ):
        failures.append("alphabetic words")
    lowered = {w.lower().strip(".,!?;:\"'") for w in words}
    if sum(sw in lowered for sw in STOP_WORDS) < rules.min_stop_words:
        failures.append("stop words")
    if top_ngram_char_fraction(words, 2) > rules.max_top_2gram_chars:
        failures.append("repeated 2-gram")
    return failures


def top_ngram_char_fraction(words: list[str], n: int) -> float:
    """Characters covered by the most frequent word n-gram, over all characters (Table A1)."""
    if len(words) < n:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    if count < 2:
        return 0.0
    return count * sum(len(w) for w in gram) / sum(len(w) for w in words)


# ---------------------------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace: small edits should not matter."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def exact_duplicates(docs: list[str]) -> list[int]:
    """Indices of documents whose normalized text was already seen (the first copy is kept)."""
    seen: set[str] = set()
    drop = []
    for i, doc in enumerate(docs):
        digest = hashlib.sha1(normalize(doc).encode()).hexdigest()
        if digest in seen:
            drop.append(i)
        seen.add(digest)
    return drop


def shingles(text: str, n: int = 5) -> set[str]:
    """Overlapping word n-grams of the normalized text."""
    words = normalize(text).split()
    return {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _stable_hash(item: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(item.encode(), digest_size=8).digest(), "little"
    )


class MinHasher:
    """Signature = for each of ``num_perm`` random hash functions h_i(x) = (a_i x + b_i) mod p,
    the minimum of h_i over the document's shingles. P(two signatures agree at i) equals the
    Jaccard similarity of the two shingle sets."""

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE_PRIME, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, MERSENNE_PRIME, size=num_perm, dtype=np.int64)

    def signature(self, items: set[str]) -> np.ndarray:
        x = np.array([_stable_hash(s) % MERSENNE_PRIME for s in items], dtype=np.int64)
        hashed = (x[:, None] * self.a[None, :] + self.b[None, :]) % MERSENNE_PRIME
        return hashed.min(axis=0)


def minhash_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


def lsh_candidate_probability(similarity: float, bands: int, rows: int) -> float:
    """P(two documents share at least one band) = 1 - (1 - s^r)^b: an S-curve in s."""
    return 1 - (1 - similarity**rows) ** bands


def lsh_candidate_pairs(
    signatures: np.ndarray, bands: int, rows: int
) -> set[tuple[int, int]]:
    """Split each signature into ``bands`` bands of ``rows`` values; documents that agree on
    a whole band land in the same bucket and become candidate pairs."""
    if signatures.shape[1] != bands * rows:
        raise ValueError("signature length must equal bands * rows")
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        for i, sig in enumerate(signatures[:, band * rows : (band + 1) * rows]):
            buckets.setdefault(sig.tobytes(), []).append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


@dataclass
class NearDuplicateResult:
    drop: list[int]
    candidates: int
    confirmed: int
    clusters: list[list[int]] = field(default_factory=list)


def near_duplicates(
    docs: list[str],
    threshold: float = 0.7,
    num_perm: int = 128,
    bands: int = 32,
    ngram: int = 5,
    seed: int = 0,
) -> NearDuplicateResult:
    """MinHash + LSH candidates, confirmed with the exact Jaccard similarity, merged into
    clusters with union-find; every cluster keeps its first document."""
    sets = [shingles(doc, ngram) for doc in docs]
    hasher = MinHasher(num_perm, seed)
    signatures = np.stack([hasher.signature(s) for s in sets])
    candidates = lsh_candidate_pairs(signatures, bands, num_perm // bands)
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    confirmed = 0
    for i, j in candidates:
        if jaccard(sets[i], sets[j]) >= threshold:
            confirmed += 1
            parent[max(find(i), find(j))] = min(find(i), find(j))
    groups: dict[int, list[int]] = {}
    for i in range(len(docs)):
        groups.setdefault(find(i), []).append(i)
    clusters = [g for g in groups.values() if len(g) > 1]
    drop = sorted(i for g in clusters for i in g[1:])
    return NearDuplicateResult(drop, len(candidates), confirmed, clusters)


# ---------------------------------------------------------------------------------------------
# A hashed bag-of-n-grams classifier (the fastText-style quality filter of DCLM, simplified)
# ---------------------------------------------------------------------------------------------


def hashed_ngram_ids(text: str, num_buckets: int, max_n: int = 2) -> list[int]:
    words = normalize(text).split()
    grams = [
        " ".join(words[i : i + n])
        for n in range(1, max_n + 1)
        for i in range(len(words) - n + 1)
    ]
    return [_stable_hash(g) % num_buckets for g in grams] or [0]


class BagOfNgramsClassifier(nn.Module):
    """Average of learned n-gram vectors -> one logit ("looks like the reference data")."""

    def __init__(self, num_buckets: int = 2**18, dim: int = 16) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.EmbeddingBag(num_buckets, dim, mode="mean")
        self.out = nn.Linear(dim, 1)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(ids, offsets)).squeeze(-1)

    def batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        lists = [hashed_ngram_ids(t, self.num_buckets) for t in texts]
        offsets = torch.tensor([0] + [len(x) for x in lists[:-1]]).cumsum(0)
        return torch.tensor([i for x in lists for i in x]), offsets

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        """P(reference-like) for each text."""
        return torch.sigmoid(self(*self.batch(texts))).numpy()


def train_quality_classifier(
    positives: list[str],
    negatives: list[str],
    epochs: int = 5,
    lr: float = 0.05,
    seed: int = 0,
) -> BagOfNgramsClassifier:
    torch.manual_seed(seed)
    model = BagOfNgramsClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = [(t, 1.0) for t in positives] + [(t, 0.0) for t in negatives]
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(data)
        for start in range(0, len(data), 64):
            chunk = data[start : start + 64]
            logits = model(*model.batch([t for t, _ in chunk]))
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.tensor([y for _, y in chunk])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


# ---------------------------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------------------------


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def contaminated(test_docs: list[str], train_docs: list[str], n: int = 13) -> list[int]:
    """Test documents sharing at least one word n-gram with the training data (GPT-3 used
    n = 13)."""
    train_grams: set[tuple[str, ...]] = set()
    for doc in train_docs:
        train_grams |= word_ngrams(doc, n)
    return [i for i, doc in enumerate(test_docs) if word_ngrams(doc, n) & train_grams]


# ---------------------------------------------------------------------------------------------
# A web-like corpus to clean: real stories plus the usual kinds of junk
# ---------------------------------------------------------------------------------------------

BOILERPLATE = [
    "Home | About us | Contact | Privacy policy | Terms of use | Login",
    "Click here to subscribe to our newsletter!",
    "Copyright 2024. All rights reserved.",
    "Share this: Facebook Twitter Email Print",
]
SPAM_WORDS = [
    "cheap",
    "best",
    "price",
    "buy",
    "online",
    "discount",
    "shoes",
    "free",
    "deal",
]


def _edit_words(text: str, rng: random.Random, fraction: float) -> str:
    words = text.split()
    for _ in range(max(1, int(len(words) * fraction))):
        words[rng.randrange(len(words))] = rng.choice(
            ["very", "big", "small", "happy", "the"]
        )
    return " ".join(words)


def _junk(rng: random.Random, kind: str) -> str:
    if kind == "spam":
        return " ".join(rng.choice(SPAM_WORDS) for _ in range(rng.randint(60, 200)))
    if kind == "menu":
        return "\n".join(rng.choice(BOILERPLATE) for _ in range(rng.randint(5, 15)))
    if kind == "table":
        rows = [
            " ".join(str(rng.randint(0, 9999)) for _ in range(6)) for _ in range(20)
        ]
        return "\n".join(rows)
    if kind == "list":
        items = [
            f"• {rng.choice(SPAM_WORDS)} {rng.choice(SPAM_WORDS)}" for _ in range(30)
        ]
        return "\n".join(items)
    if kind == "gibberish":
        letters = "abcdefghijklmnopqrstuvwxyz"
        return " ".join(
            "".join(rng.choice(letters) for _ in range(rng.randint(1, 14)))
            for _ in range(120)
        )
    raise ValueError(kind)


JUNK_KINDS = ("spam", "menu", "table", "list", "gibberish")


def make_web_corpus(
    stories: list[str],
    num_exact: int = 150,
    num_near: int = 150,
    num_junk_each: int = 60,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Mix clean stories with exact copies, near-copies (a few words changed and boilerplate
    added) and five kinds of junk. Returns (documents, labels).

    The clean stories come first and the rest is shuffled after them, so "keep the first copy"
    keeps the original and the labels say exactly what a perfect filter should drop.
    """
    rng = random.Random(seed)
    extra = []
    for s in rng.sample(stories, num_exact):
        extra.append(("  " + s.replace(". ", ".  ") + "\n", "exact duplicate"))
    for s in rng.sample(stories, num_near):
        edited = _edit_words(s, rng, 0.03)
        extra.append(
            (
                f"{rng.choice(BOILERPLATE)}\n{edited}\n{rng.choice(BOILERPLATE)}",
                "near duplicate",
            )
        )
    for kind in JUNK_KINDS:
        extra += [(_junk(rng, kind), f"junk: {kind}") for _ in range(num_junk_each)]
    rng.shuffle(extra)
    docs = [(s, "clean") for s in stories] + extra
    return [d for d, _ in docs], [label for _, label in docs]
