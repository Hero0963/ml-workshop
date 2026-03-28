# scripts/train_tokenizer.py
"""
Train a BPE tokenizer from scratch on a text file.
Implements both BasicTokenizer and RegexTokenizer.

Usage:
    uv run python scripts/train_tokenizer.py
"""

import os
import time

# ---------------------------------------------------------------------------
# Helper functions (from minBPE)
# ---------------------------------------------------------------------------


def get_stats(ids, counts=None):
    """Count consecutive pairs in a list of integers."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace all consecutive occurrences of pair with idx."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# ---------------------------------------------------------------------------
# BasicTokenizer
# ---------------------------------------------------------------------------


class BasicTokenizer:
    """Minimal byte-level BPE tokenizer."""

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}  # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                token_str = vocab[idx].decode("utf-8", errors="replace")
                # Use ascii() to avoid Windows console encoding issues
                print(
                    f"  merge {i + 1}/{num_merges}: "
                    f"{pair} -> {idx} ({ascii(token_str)}) "
                    f"count={stats[pair]}"
                )

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Find training text
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    text_path = os.path.join(
        project_dir, "references", "minbpe", "tests", "taylorswift.txt"
    )

    if not os.path.exists(text_path):
        print(f"Text file not found: {text_path}")
        print("Falling back to a demo string...")
        text = "The quick brown fox jumps over the lazy dog. " * 200
    else:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

    print(f"Training text: {len(text):,} characters")
    print(f"UTF-8 bytes:   {len(text.encode('utf-8')):,}")
    print()

    # Train tokenizer
    vocab_size = 512
    print(f"Training BasicTokenizer with vocab_size={vocab_size}...")
    print(f"  (256 byte tokens + {vocab_size - 256} merges)")
    print()

    tokenizer = BasicTokenizer()
    t0 = time.time()
    tokenizer.train(text, vocab_size, verbose=True)
    t1 = time.time()
    print(f"\nTraining took {t1 - t0:.2f}s")

    # Test encode/decode
    test_text = text[:200]
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print("\n--- Encode/Decode Test ---")
    print(f"Original:  {len(test_text)} chars")
    print(f"Encoded:   {len(encoded)} tokens")
    print(f"Decoded OK: {decoded == test_text}")
    print(f"Compression ratio: {len(test_text.encode('utf-8')) / len(encoded):.2f}x")

    # Show some learned tokens
    print("\n--- Learned Tokens (last 20) ---")
    for idx in range(max(256, vocab_size - 20), vocab_size):
        token_bytes = tokenizer.vocab[idx]
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"  {idx}: {ascii(token_str)} ({len(token_bytes)} bytes)")


if __name__ == "__main__":
    main()
