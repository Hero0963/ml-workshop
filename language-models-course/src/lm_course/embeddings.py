# src/lm_course/embeddings.py
"""Text embedding models (lesson 13): pooling, contrastive training, Matryoshka, evaluation.

A text embedding model maps a text to one vector so that related texts are close. Here the
backbone is the GPT from lesson 06; training uses InfoNCE with in-batch negatives on pairs of
spans cut from the same story (the "independent cropping" of Contriever, Izacard et al. 2021).
"""

import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from lm_course.model import GPT
from lm_course.tokenizer import BPETokenizer

BM25_K1 = 1.5
BM25_B = 0.75


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average of the hidden states of the real (non-padding) tokens."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the last real token: with causal attention it is the only position
    that has seen the whole text (the choice of E5-Mistral and Qwen3-Embedding)."""
    last = mask.long().sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class TextEncoder(nn.Module):
    """GPT trunk + pooling (+ optional linear projection), unit-normalized outputs."""

    def __init__(
        self, backbone: GPT, pooling: str = "mean", out_dim: int | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        d = backbone.config.d_model
        self.proj = nn.Linear(d, out_dim, bias=False) if out_dim else nn.Identity()

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padding_mask = mask.bool() if not self.backbone.config.causal else None
        hidden = self.backbone.hidden_states(ids, padding_mask=padding_mask)
        pooled = (
            mean_pool(hidden, mask)
            if self.pooling == "mean"
            else last_token_pool(hidden, mask)
        )
        return F.normalize(self.proj(pooled), dim=-1)

    @torch.no_grad()
    def encode(
        self,
        tokenizer: BPETokenizer,
        texts: list[str],
        max_tokens: int,
        batch_size: int = 64,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        out = []
        for start in range(0, len(texts), batch_size):
            ids, mask = tokenize_batch(
                tokenizer, texts[start : start + batch_size], max_tokens
            )
            out.append(self(ids.to(device), mask.to(device)).cpu())
        return torch.cat(out)


def tokenize_batch(
    tokenizer: BPETokenizer, texts: list[str], max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids and a 0/1 mask; each text starts with <|bos|>."""
    bos = tokenizer.special_id("<|bos|>")
    rows = [[bos, *tokenizer.encode_ordinary(t)][:max_tokens] for t in texts]
    length = max(len(r) for r in rows)
    ids = torch.full((len(rows), length), bos, dtype=torch.long)
    mask = torch.zeros(len(rows), length, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    return ids, mask


# ---------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------


def info_nce_loss(
    queries: torch.Tensor,
    docs: torch.Tensor,
    temperature: float = 0.05,
    symmetric: bool = True,
) -> torch.Tensor:
    """Row i of ``queries`` should match row i of ``docs``; every other row in the batch is a
    negative. Cross-entropy over cos(q_i, d_j) / temperature (van den Oord et al. 2018)."""
    logits = queries @ docs.T / temperature
    labels = torch.arange(len(queries), device=queries.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = (loss + F.cross_entropy(logits.T, labels)) / 2
    return loss


def matryoshka_loss(
    queries: torch.Tensor,
    docs: torch.Tensor,
    dims: list[int],
    temperature: float = 0.05,
) -> torch.Tensor:
    """Average InfoNCE over nested prefixes of the embedding (Kusupati et al. 2022), so the
    first 16, 32, ... dimensions are each a usable embedding on their own."""
    losses = [
        info_nce_loss(
            F.normalize(queries[:, :k], dim=-1),
            F.normalize(docs[:, :k], dim=-1),
            temperature,
        )
        for k in dims
    ]
    return torch.stack(losses).mean()


def alignment_and_uniformity(
    x: torch.Tensor, y: torch.Tensor, t: float = 2.0
) -> tuple[float, float]:
    """Wang & Isola (2020): alignment = E ||x - y||^2 over positive pairs (lower = positives
    closer); uniformity = log E exp(-t ||u - v||^2) over all pairs (lower = spread out)."""
    alignment = (x - y).pow(2).sum(dim=-1).mean().item()
    distances = torch.pdist(x).pow(2)
    uniformity = torch.log(torch.exp(-t * distances).mean()).item()
    return alignment, uniformity


# ---------------------------------------------------------------------------------------------
# Data: two views of the same story
# ---------------------------------------------------------------------------------------------


def split_halves(text: str) -> tuple[str, str]:
    """First and second half of a text, cut at a sentence boundary near the middle."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) < 2:
        words = text.split()
        return " ".join(words[: len(words) // 2]), " ".join(words[len(words) // 2 :])
    cut = max(1, len(sentences) // 2)
    return " ".join(sentences[:cut]), " ".join(sentences[cut:])


def random_crops(
    text: str, rng: random.Random, min_frac: float = 0.2, max_frac: float = 0.5
) -> tuple[str, str]:
    """Two independent random word spans of the same text: a positive pair with no labels."""
    words = text.split()
    crops = []
    for _ in range(2):
        n = max(1, int(len(words) * rng.uniform(min_frac, max_frac)))
        start = rng.randrange(0, max(1, len(words) - n + 1))
        crops.append(" ".join(words[start : start + n]))
    return crops[0], crops[1]


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------


@dataclass
class ContrastiveResult:
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train_contrastive(
    encoder: TextEncoder,
    tokenizer: BPETokenizer,
    texts: list[str],
    num_steps: int,
    batch_size: int = 64,
    max_tokens: int = 96,
    lr: float = 3e-4,
    temperature: float = 0.05,
    matryoshka_dims: list[int] | None = None,
    seed: int = 0,
    log_every: int = 100,
) -> ContrastiveResult:
    """InfoNCE on random-crop pairs, a fresh batch of texts every step."""
    rng = random.Random(seed)
    device = next(encoder.parameters()).device
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.01)
    result = ContrastiveResult()
    encoder.train()
    start = time.perf_counter()
    for step in range(num_steps):
        batch = rng.sample(texts, batch_size)
        pairs = [random_crops(t, rng) for t in batch]
        qa, qm = tokenize_batch(tokenizer, [a for a, _ in pairs], max_tokens)
        da, dm = tokenize_batch(tokenizer, [b for _, b in pairs], max_tokens)
        q = encoder(qa.to(device), qm.to(device))
        d = encoder(da.to(device), dm.to(device))
        if matryoshka_dims:
            loss = matryoshka_loss(q, d, matryoshka_dims, temperature)
        else:
            loss = info_nce_loss(q, d, temperature)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        result.losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            logger.info(
                f"step {step + 1}/{num_steps}: loss {np.mean(result.losses[-log_every:]):.3f}"
            )
    result.seconds = time.perf_counter() - start
    encoder.eval()
    return result


# ---------------------------------------------------------------------------------------------
# Evaluation and baselines
# ---------------------------------------------------------------------------------------------


def retrieval_metrics(
    scores: torch.Tensor | np.ndarray, ks: tuple[int, ...] = (1, 10)
) -> dict[str, float]:
    """``scores[i, j]`` = similarity of query i and document j; the right document is j = i.
    Returns recall@k (the right document is in the top k) and MRR (mean 1 / rank)."""
    scores = torch.as_tensor(scores, dtype=torch.float64)
    correct = scores.diagonal()[:, None]
    ranks = (scores > correct).sum(dim=1) + 1
    metrics = {f"recall@{k}": (ranks <= k).double().mean().item() for k in ks}
    metrics["mrr"] = (1.0 / ranks.double()).mean().item()
    return metrics


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def bm25_scores(
    queries: list[str], docs: list[str], k1: float = BM25_K1, b: float = BM25_B
) -> np.ndarray:
    """Okapi BM25: sum over query words of idf(w) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avg|d|)).
    The classic lexical baseline every embedding model is compared against."""
    doc_words = [Counter(_words(d)) for d in docs]
    lengths = np.array([sum(c.values()) for c in doc_words], dtype=np.float64)
    avg = lengths.mean()
    df = Counter(w for c in doc_words for w in c)
    n = len(docs)
    idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
    vocab = {w: i for i, w in enumerate(df)}
    # weight[w, j] = the BM25 term weight of word w in document j
    weights = np.zeros((len(vocab), n))
    for j, counts in enumerate(doc_words):
        norm = k1 * (1 - b + b * lengths[j] / avg)
        for w, tf in counts.items():
            weights[vocab[w], j] = idf[w] * tf * (k1 + 1) / (tf + norm)
    scores = np.zeros((len(queries), n))
    for i, q in enumerate(queries):
        rows = [vocab[w] for w in set(_words(q)) if w in vocab]
        if rows:
            scores[i] = weights[rows].sum(axis=0)
    return scores


def average_word_vectors(
    texts: list[str], vectors: np.ndarray, index: dict[str, int]
) -> torch.Tensor:
    """Mean of the (unit) word2vec vectors of the known words: a bag-of-embeddings baseline."""
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(texts), vectors.shape[1]))
    for i, text in enumerate(texts):
        ids = [index[w] for w in _words(text) if w in index]
        if ids:
            out[i] = unit[ids].mean(axis=0)
    return F.normalize(torch.from_numpy(out).float(), dim=-1)
