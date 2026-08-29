# src/core/tests/rl/test_generate_dataset_v2.py
"""Tests for the dataset digests that make an RL training set identifiable.

Re-running the builder does not reproduce a dataset bit-for-bit — `generate_puzzle` gives
up on wall-clock time — so a dataset is identified by its content digest rather than by the
command that made it. These tests pin the two properties that makes possible: the digest
must ignore how Python happens to order a set or a dict, and it must not ignore the data.
"""

import json
import pickle
import random

import pytest

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.generate_dataset_v2 import (
    content_digests,
    deduplicate,
    sample_fingerprint,
    split_digest,
    verify_dataset,
)
from src.core.rl.rl_env_v2 import PuzzleSample

GENERATOR_SEED = 20260829


@pytest.fixture(scope="module")
def samples() -> list[PuzzleSample]:
    built: list[PuzzleSample] = []
    for offset in range(3):
        random.seed(GENERATOR_SEED + offset)
        result = generate_puzzle(m=4, n=4, has_walls=True, timeout_per_attempt=5.0)
        assert result is not None, "generator returned None; retry with another seed"
        puzzle, solution_path = result
        built.append(PuzzleSample(puzzle=puzzle, solution_path=solution_path))
    return built


def test_fingerprint_ignores_set_and_dict_ordering(samples: list[PuzzleSample]) -> None:
    original = samples[0]
    shuffled_walls = list(original.puzzle["walls"])
    random.Random(GENERATOR_SEED).shuffle(shuffled_walls)
    reordered = PuzzleSample(
        puzzle={
            **original.puzzle,
            "walls": set(reversed(shuffled_walls)),
            "num_map": dict(reversed(list(original.puzzle["num_map"].items()))),
        },
        solution_path=list(original.solution_path),
    )

    assert sample_fingerprint(reordered) == sample_fingerprint(original)


def test_fingerprint_notices_a_changed_wall(samples: list[PuzzleSample]) -> None:
    original = samples[0]
    tampered = PuzzleSample(
        puzzle={**original.puzzle, "walls": set()},
        solution_path=original.solution_path,
    )

    assert sample_fingerprint(tampered) != sample_fingerprint(original)


def test_fingerprint_notices_a_changed_solution(samples: list[PuzzleSample]) -> None:
    original = samples[0]
    tampered = PuzzleSample(
        puzzle=original.puzzle, solution_path=original.solution_path[::-1]
    )

    assert sample_fingerprint(tampered) != sample_fingerprint(original)


def test_split_digest_depends_on_order(samples: list[PuzzleSample]) -> None:
    """Two splits holding the same puzzles in a different order are not the same split."""
    assert split_digest(samples) != split_digest(list(reversed(samples)))


def test_verify_accepts_a_matching_dataset(samples, tmp_path) -> None:
    splits = {"train": samples[:2], "val": samples[2:], "test": []}
    _write_dataset(tmp_path, splits, content_digests(splits))

    assert verify_dataset(tmp_path) is True


def test_verify_rejects_a_tampered_dataset(samples, tmp_path) -> None:
    splits = {"train": samples[:2], "val": samples[2:], "test": []}
    recorded = content_digests(splits)
    splits["train"] = list(reversed(splits["train"]))
    _write_dataset(tmp_path, splits, recorded)

    assert verify_dataset(tmp_path) is False


def test_verify_rejects_a_dataset_built_before_digests_existed(
    samples, tmp_path
) -> None:
    _write_dataset(tmp_path, {"train": samples, "val": [], "test": []}, None)

    assert verify_dataset(tmp_path) is False


def test_deduplicate_keeps_the_first_of_each_repeat(
    samples: list[PuzzleSample],
) -> None:
    records = [
        {"sample": samples[0], "seed": 1},
        {"sample": samples[1], "seed": 2},
        {"sample": samples[0], "seed": 3},
        {"sample": samples[2], "seed": 4},
    ]

    unique, dropped = deduplicate(records)

    assert dropped == 1
    assert [record["seed"] for record in unique] == [1, 2, 4]


def test_deduplicate_makes_splits_disjoint(samples: list[PuzzleSample]) -> None:
    """4x4 at 20k put 111 puzzles in both train and test; splits must not overlap."""
    records = [{"sample": sample, "seed": i} for i, sample in enumerate(samples * 3)]

    unique, dropped = deduplicate(records)
    fingerprints = [sample_fingerprint(record["sample"]) for record in unique]

    assert dropped == len(records) - len(samples)
    assert len(set(fingerprints)) == len(fingerprints)


def _write_dataset(directory, splits, digests) -> None:
    manifest = {"total_generated": sum(len(s) for s in splits.values())}
    if digests is not None:
        manifest["content_sha256"] = digests
    with (directory / "dataset.pkl").open("wb") as handle:
        pickle.dump({"splits": splits, "manifest": manifest}, handle)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
