# tests/test_tokenizer.py
import pytest
import regex

from helpers import download_or_skip
from lm_course.tokenizer import (
    COURSE_SPLIT_PATTERN,
    GPT2_SPLIT_PATTERN,
    BPETokenizer,
    gpt2_byte_to_unicode,
    gpt2_tokenizer,
)

CORPUS = [
    "Once upon a time, there was a little dog named Max. Max liked to run.",
    'The little girl said, "I don\'t want to go!" She was 7 years old.',
    "héllo wörld, naïve café — 你好世界 🙂 it's they're we'll 12345",
] * 20


@pytest.fixture(scope="module")
def trained() -> BPETokenizer:
    return BPETokenizer.train(
        CORPUS, vocab_size=330, special_tokens=["<|bos|>", "<|end|>"]
    )


def test_vocab_size_and_ids(trained: BPETokenizer) -> None:
    assert trained.vocab_size == 330
    assert sorted(trained.vocab) == list(range(330))
    assert trained.special_tokens == {"<|bos|>": 328, "<|end|>": 329}


def test_training_stops_when_every_word_is_one_token() -> None:
    tok = BPETokenizer.train(["abc abc"], vocab_size=1000)
    assert len(tok.encode("abc abc")) == 2  # "abc" and " abc"
    assert tok.vocab_size < 1000


def test_round_trip_on_unseen_text(trained: BPETokenizer) -> None:
    text = "Ünïcödé never seen before: ∑ 漢字 🚀\n\n  tabs\tand  spaces  "
    assert trained.decode(trained.encode(text)) == text


def test_merges_compress_the_training_text(trained: BPETokenizer) -> None:
    text = CORPUS[0]
    assert len(trained.encode(text)) < len(text.encode("utf-8")) / 2


def test_every_learned_token_stays_inside_one_pre_token(trained: BPETokenizer) -> None:
    pattern = regex.compile(COURSE_SPLIT_PATTERN)
    for token_id, token in trained.vocab.items():
        if trained.is_special(token_id):
            continue
        text = token.decode("utf-8", errors="ignore")
        if text:
            assert len(pattern.findall(text)) == 1, text


def test_classic_example_first_merge_is_the_most_frequent_pair() -> None:
    tok = BPETokenizer.train(["aaabdaaabac"], vocab_size=259)
    a = ord("a")
    assert tok.merges[(a, a)] == 256  # "aa" occurs 4 times, more than any other pair
    assert tok.decode(tok.encode("aaabdaaabac")) == "aaabdaaabac"


def test_special_tokens_only_when_allowed(trained: BPETokenizer) -> None:
    text = "hi<|bos|>there"
    with_special = trained.encode(text, allowed_special=True)
    assert trained.special_id("<|bos|>") in with_special
    assert trained.special_id("<|bos|>") not in trained.encode(text)
    assert trained.decode(with_special) == text


def test_digits_are_single_tokens(trained: BPETokenizer) -> None:
    ids = trained.encode("12345")
    assert [trained.decode([i]) for i in ids] == ["1", "2", "3", "4", "5"]


def test_save_and_load(tmp_path, trained: BPETokenizer) -> None:  # noqa: ANN001
    path = tmp_path / "tok.json"
    trained.save(path)
    loaded = BPETokenizer.load(path)
    text = " ".join(CORPUS[:3])
    assert loaded.encode(text, allowed_special=True) == trained.encode(
        text, allowed_special=True
    )


def test_token_byte_lengths(trained: BPETokenizer) -> None:
    lengths = trained.token_byte_lengths()
    assert lengths[trained.special_id("<|bos|>")] == 0
    ids = trained.encode(CORPUS[2])
    assert sum(lengths[i] for i in ids) == len(CORPUS[2].encode("utf-8"))


def test_gpt2_byte_table_is_a_printable_bijection() -> None:
    table = gpt2_byte_to_unicode()
    assert sorted(table) == list(range(256))
    assert len(set(table.values())) == 256
    assert all(ch.isprintable() and not ch.isspace() for ch in table.values())
    assert table[ord(" ")] == "Ġ"


def test_gpt2_pattern_splits_like_the_paper() -> None:
    assert regex.findall(GPT2_SPLIT_PATTERN, "Hello world's 123!!") == [
        "Hello",
        " world",
        "'s",
        " 123",
        "!!",
    ]


@pytest.mark.network
def test_gpt2_tokenizer_matches_known_ids() -> None:
    tok = download_or_skip(gpt2_tokenizer)
    # reference ids from OpenAI's tokenizer (tiktoken "gpt2"), checked on 2026-09-25
    assert tok.encode("Hello world") == [15496, 995]
    assert tok.encode("The capital of France is") == [464, 3139, 286, 4881, 318]
    assert tok.encode("a<|endoftext|>b", allowed_special=True) == [64, 50256, 65]
