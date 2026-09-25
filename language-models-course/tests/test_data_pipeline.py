# tests/test_data_pipeline.py
import random

import numpy as np
import pytest

from lm_course.data_pipeline import (
    MinHasher,
    contaminated,
    exact_duplicates,
    gopher_failures,
    jaccard,
    lsh_candidate_pairs,
    lsh_candidate_probability,
    make_web_corpus,
    minhash_similarity,
    near_duplicates,
    shingles,
    train_quality_classifier,
)

STORY = (
    "Once upon a time there was a little girl named Lily. She liked to play with her dog in "
    "the park. One day the dog ran after a red ball and got lost. Lily looked everywhere and "
    "was very sad. Then she heard a bark behind the big tree. It was her dog, and he had the "
    "ball in his mouth. Lily hugged him and they walked home together, happy to be friends."
)


def test_a_normal_story_passes_the_gopher_rules() -> None:
    assert gopher_failures(STORY) == []


@pytest.mark.parametrize(
    "text, rule",
    [
        ("Too short to keep.", "word count"),
        (" ".join(["cheap shoes"] * 60), "stop words"),
        (" ".join(["#tag"] * 60) + " the and of", "symbol-to-word ratio"),
        ("\n".join(["• the item and more"] * 60), "bullet lines"),
        (" ".join(["1234 5678"] * 40) + " the and", "alphabetic words"),
        ("\n".join([STORY.split(".")[0]] * 12), "duplicate lines"),
    ],
)
def test_each_rule_catches_its_kind_of_junk(text: str, rule: str) -> None:
    assert rule in gopher_failures(text)


def test_exact_duplicates_ignore_case_whitespace_and_punctuation() -> None:
    docs = ["Hello, World!", "hello world", "HELLO   world.", "something else"]
    assert exact_duplicates(docs) == [1, 2]


def test_minhash_estimates_jaccard() -> None:
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(400)]
    a = set(rng.sample(vocab, 200))
    b = set(list(a)[:150]) | set(rng.sample(vocab, 80))
    hasher = MinHasher(num_perm=512, seed=1)
    est = minhash_similarity(hasher.signature(a), hasher.signature(b))
    assert est == pytest.approx(jaccard(a, b), abs=0.05)


def test_lsh_probability_is_an_s_curve() -> None:
    low, mid, high = (
        lsh_candidate_probability(s, bands=20, rows=5) for s in (0.2, 0.55, 0.9)
    )
    assert low < 0.01 and high > 0.99 and 0.2 < mid < 0.8


def test_lsh_finds_pairs_that_share_a_band() -> None:
    sigs = np.array([[1, 2, 3, 4], [1, 2, 9, 9], [7, 7, 3, 4], [5, 6, 8, 0]])
    assert lsh_candidate_pairs(sigs, bands=2, rows=2) == {(0, 1), (0, 2)}


def test_near_duplicates_are_clustered_and_distinct_texts_kept() -> None:
    edited = STORY.replace("red ball", "blue ball").replace("very sad", "so sad")
    other = "The weather report says it will rain tomorrow in the northern mountains."
    result = near_duplicates([STORY, other, edited, STORY + " The end."], threshold=0.7)
    assert result.drop == [2, 3]
    assert jaccard(shingles(STORY), shingles(edited)) > 0.7


def test_quality_classifier_separates_prose_from_spam() -> None:
    rng = random.Random(0)
    words = STORY.split()
    prose = [" ".join(rng.sample(words, 40)) for _ in range(200)]
    spam = [
        " ".join(rng.choice(["cheap", "buy", "deal", "free", "now"]) for _ in range(40))
        for _ in range(200)
    ]
    model = train_quality_classifier(prose[:150], spam[:150], epochs=3)
    assert model.score(prose[150:]).mean() > 0.8
    assert model.score(spam[150:]).mean() < 0.2


def test_contamination_finds_shared_ngrams() -> None:
    train = [STORY]
    test = [
        STORY.split(".")[1] + " and then more new words here",
        "entirely different words " * 5,
    ]
    assert contaminated(test, train, n=8) == [0]


def test_web_corpus_puts_originals_first() -> None:
    stories = [f"{STORY} Number {i}." for i in range(50)]
    docs, labels = make_web_corpus(stories, num_exact=5, num_near=5, num_junk_each=2)
    assert labels[:50] == ["clean"] * 50
    assert len(docs) == 50 + 5 + 5 + 10
    assert sorted(exact_duplicates(docs)) == [
        i for i, lab in enumerate(labels) if lab == "exact duplicate"
    ]
