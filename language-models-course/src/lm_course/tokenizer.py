# src/lm_course/tokenizer.py
"""Byte-level BPE (lesson 03): training, encoding, decoding, special tokens, GPT-2's vocabulary.

A token is a byte string. Training starts from the 256 single bytes and repeatedly merges the
most frequent adjacent pair. Encoding replays the merges in the order they were learned.
Text is first cut into "pre-tokens" by a regex, and merges never cross a pre-token boundary.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex

# GPT-2 (2019): contractions, letters, numbers, other symbols, whitespace.
GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
# GPT-4 (cl100k_base): case-insensitive contractions, numbers in groups of at most 3 digits.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)
# This course: the GPT-4 pattern with every digit on its own, which makes arithmetic easier
# for small models (lesson 03 §2.4; nanochat uses groups of at most 2 digits).
COURSE_SPLIT_PATTERN = GPT4_SPLIT_PATTERN.replace(r"\p{N}{1,3}", r"\p{N}")

NUM_BYTES = 256


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE with optional special tokens.

    ``byte_to_id[b]`` is the id of the single byte ``b`` (GPT-2 does not use the identity
    mapping), ``merges[(a, b)]`` is the id of the token made by merging ``a`` and ``b``.
    Merged ids are assigned in training order, so a smaller id means an earlier merge.
    """

    def __init__(
        self,
        byte_to_id: list[int],
        merges: dict[tuple[int, int], int],
        pattern: str = COURSE_SPLIT_PATTERN,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.byte_to_id = byte_to_id
        self.merges = merges
        self.pattern = pattern
        self._compiled = regex.compile(pattern)
        self.special_tokens = dict(special_tokens or {})
        self.vocab: dict[int, bytes] = {
            byte_to_id[b]: bytes([b]) for b in range(NUM_BYTES)
        }
        for (a, b), new_id in sorted(merges.items(), key=lambda item: item[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        for name, token_id in self.special_tokens.items():
            self.vocab[token_id] = name.encode("utf-8")
        self._special_ids = set(self.special_tokens.values())
        self._special_pattern = (
            regex.compile(
                "(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")"
            )
            if self.special_tokens
            else None
        )
        self._cache: dict[bytes, list[int]] = {}

    # ---- training ---------------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        pattern: str = COURSE_SPLIT_PATTERN,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """Learn ``vocab_size - 256 - len(special_tokens)`` merges from ``texts``.

        Pre-tokens are counted once, so the work per merge depends on the number of distinct
        pre-tokens, not on the corpus size. Ties between equally frequent pairs go to the pair
        whose bytes compare larger, which makes training deterministic.
        """
        special_tokens = special_tokens or []
        num_merges = vocab_size - NUM_BYTES - len(special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size {vocab_size} is smaller than 256 + special tokens"
            )

        compiled = regex.compile(pattern)
        splitter = (
            regex.compile("|".join(regex.escape(s) for s in special_tokens))
            if special_tokens
            else None
        )
        word_counts: Counter[bytes] = Counter()
        for text in texts:
            pieces = splitter.split(text) if splitter else [text]
            for piece in pieces:
                word_counts.update(m.encode("utf-8") for m in compiled.findall(piece))

        words = [list(word) for word in word_counts]
        freqs = list(word_counts.values())
        vocab = {i: bytes([i]) for i in range(NUM_BYTES)}
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for w, (ids, freq) in enumerate(zip(words, freqs)):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(w)

        merges: dict[tuple[int, int], int] = {}
        for new_id in range(NUM_BYTES, NUM_BYTES + num_merges):
            if not pair_counts:
                break
            best = max(
                pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]])
            )
            if pair_counts[best] <= 0:
                break
            merges[best] = new_id
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]
            for w in list(pair_to_words[best]):
                old = words[w]
                new = _merge_pair(old, best, new_id)
                freq = freqs[w]
                for pair in zip(old, old[1:]):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                for pair in zip(new, new[1:]):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(w)
                words[w] = new
            del pair_to_words[best]

        next_id = NUM_BYTES + len(merges)
        specials = {name: next_id + i for i, name in enumerate(special_tokens)}
        return cls(list(range(NUM_BYTES)), merges, pattern, specials)

    # ---- encoding and decoding --------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = [self.byte_to_id[b] for b in chunk]
        while len(ids) >= 2:
            # the earliest-learned merge among the adjacent pairs is applied first
            pair = min(
                zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf"))
            )
            if pair not in self.merges:
                break
            ids = _merge_pair(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ``text``, treating special-token strings as ordinary text."""
        ids = []
        for match in self._compiled.findall(text):
            ids.extend(self._encode_chunk(match.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: bool = False) -> list[int]:
        """Encode ``text``; with ``allowed_special`` the special-token strings become their ids."""
        if not allowed_special or self._special_pattern is None:
            return self.encode_ordinary(text)
        ids = []
        for piece in self._special_pattern.split(text):
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
            elif piece:
                ids.extend(self.encode_ordinary(piece))
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        # Invalid UTF-8 can come out of a model (e.g. half of a multi-byte character).
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def special_id(self, name: str) -> int:
        return self.special_tokens[name]

    def is_special(self, token_id: int) -> bool:
        return token_id in self._special_ids

    def token_byte_lengths(self) -> list[int]:
        """Bytes per token id, 0 for special tokens: the denominator of bits-per-byte."""
        lengths = [0] * self.vocab_size
        for token_id, token in self.vocab.items():
            lengths[token_id] = 0 if token_id in self._special_ids else len(token)
        return lengths

    # ---- persistence ------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        payload = {
            "pattern": self.pattern,
            "byte_to_id": self.byte_to_id,
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        merges = {(a, b): new_id for a, b, new_id in payload["merges"]}
        return cls(
            payload["byte_to_id"], merges, payload["pattern"], payload["special_tokens"]
        )

    @classmethod
    def from_gpt2_files(cls, vocab_json: Path, merges_txt: Path) -> "BPETokenizer":
        """Rebuild GPT-2's tokenizer from its released ``vocab.json`` and ``merges.txt``.

        GPT-2 stores tokens as printable strings: every byte is first mapped to a visible
        unicode character (``gpt2_byte_to_unicode``), so a space shows up as "Ġ".
        """
        unicode_to_byte = {ch: b for b, ch in gpt2_byte_to_unicode().items()}
        encoder: dict[str, int] = json.loads(vocab_json.read_text(encoding="utf-8"))
        byte_to_id = [encoder[gpt2_byte_to_unicode()[b]] for b in range(NUM_BYTES)]
        lines = merges_txt.read_text(encoding="utf-8").splitlines()
        merges = {}
        for line in lines:
            if not line or line.startswith("#version"):
                continue
            left, right = line.split(" ")
            merged = encoder[left + right]
            merges[(encoder[left], encoder[right])] = merged
        specials = {"<|endoftext|>": encoder["<|endoftext|>"]}
        tokenizer = cls(byte_to_id, merges, GPT2_SPLIT_PATTERN, specials)
        # sanity check: every vocabulary entry decodes to the bytes its string spells
        for token, token_id in encoder.items():
            if token != "<|endoftext|>":
                expected = bytes(unicode_to_byte[ch] for ch in token)
                assert tokenizer.vocab[token_id] == expected, token
        return tokenizer


def gpt2_byte_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable character table.

    Printable Latin-1 bytes map to themselves; the other 68 bytes (control characters, space,
    ...) are moved to code points 256 and up, in increasing byte order.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table = {b: chr(b) for b in printable}
    shift = 0
    for b in range(NUM_BYTES):
        if b not in table:
            table[b] = chr(NUM_BYTES + shift)
            shift += 1
    return table


def gpt2_tokenizer() -> BPETokenizer:
    """GPT-2's tokenizer, downloaded from the Hugging Face mirror of the original release."""
    from lm_course.data import gpt2_file

    return BPETokenizer.from_gpt2_files(
        gpt2_file("vocab.json"), gpt2_file("merges.txt")
    )
