# src/core/tests/rl/test_eval_set.py
"""The frozen eval set: JSON round trip, digest check, and the tracked copy itself.

The tracked copy is what makes a published best-of-N number re-measurable by someone who
does not have `datasets/`, so the last test pins it: if its content ever drifts from the
manifest, every score measured on it silently becomes a score on different puzzles.
"""

import json
import pickle
from pathlib import Path

import pytest

from src.core.rl.eval_set import (
    EVAL_SET_ROOT,
    MANIFEST_FILENAME,
    SAMPLES_FILENAME,
    export_split,
    load_eval_set,
    matches_source,
    record_to_sample,
    sample_to_record,
)
from src.core.rl.generate_dataset_v2 import split_digest
from src.core.rl.rl_env_v2 import PuzzleSample
from src.core.utils import parse_puzzle_layout

TRACKED_EVAL_SET = EVAL_SET_ROOT / "seed20300000_n20000_456_test"
TRACKED_PUZZLES_PER_SIZE = {"4": 1931, "5": 2001, "6": 2000}
DATASET_NAME = "tiny"


def _sample(layout: list[list[str]], walls: set, path: list) -> PuzzleSample:
    puzzle = parse_puzzle_layout(layout)
    puzzle["walls"] = walls
    return PuzzleSample(puzzle, path)


def _samples() -> list[PuzzleSample]:
    # A wall stored as (lower cell, upper cell) -- the reverse of sorted order -- so the
    # round trip has to keep endpoint order rather than normalise it.
    return [
        _sample(
            [["01", "  ", "  "], ["  ", "  ", "02"]],
            {((1, 1), (0, 1))},
            [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 1)],
        ),
        _sample(
            [["01", "03"], ["  ", "02"]],
            set(),
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        ),
    ]


def _write_dataset(root: Path, samples: list[PuzzleSample], digest: str | None) -> Path:
    dataset_dir = root / DATASET_NAME
    dataset_dir.mkdir(parents=True)
    with (dataset_dir / "dataset.pkl").open("wb") as handle:
        pickle.dump({"splits": {"test": samples}}, handle)
    manifest = {"content_sha256": {"test": digest}} if digest else {}
    (dataset_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_dir


def test_a_record_rebuilds_the_same_sample() -> None:
    for sample in _samples():
        rebuilt = record_to_sample(json.loads(json.dumps(sample_to_record(sample))))
        assert rebuilt == sample


def test_export_then_load_gives_the_source_back(tmp_path: Path) -> None:
    samples = _samples()
    dataset_dir = _write_dataset(tmp_path / "datasets", samples, split_digest(samples))

    manifest = export_split(dataset_dir, "test", tmp_path / "out")
    loaded_manifest, loaded = load_eval_set(tmp_path / "out")

    assert loaded == samples
    assert loaded_manifest == manifest
    assert manifest["puzzles_per_size"] == {"2": 2}
    assert matches_source(tmp_path / "out", tmp_path / "datasets")


def test_a_dataset_without_a_digest_is_refused(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path, _samples(), digest=None)
    with pytest.raises(ValueError, match="no content digest"):
        export_split(dataset_dir, "test", tmp_path / "out")


def test_an_edited_eval_set_is_refused(tmp_path: Path) -> None:
    samples = _samples()
    dataset_dir = _write_dataset(tmp_path / "datasets", samples, split_digest(samples))
    export_split(dataset_dir, "test", tmp_path / "out")

    lines = (
        (tmp_path / "out" / SAMPLES_FILENAME).read_text(encoding="utf-8").splitlines()
    )
    (tmp_path / "out" / SAMPLES_FILENAME).write_text(
        "\n".join(reversed(lines)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        load_eval_set(tmp_path / "out")


@pytest.mark.skipif(
    not TRACKED_EVAL_SET.exists(),
    reason="rl_eval_sets/ lives in the repo but is not copied into the Docker image",
)
def test_the_tracked_eval_set_matches_its_manifest() -> None:
    manifest, samples = load_eval_set(TRACKED_EVAL_SET)
    per_size: dict[str, int] = {}
    for sample in samples:
        size = str(sample.puzzle["grid_size"][0])
        per_size[size] = per_size.get(size, 0) + 1

    assert per_size == TRACKED_PUZZLES_PER_SIZE == manifest["puzzles_per_size"]
    assert manifest["dataset"] == "seed20300000_n20000_456"
    assert manifest["split"] == "test"
