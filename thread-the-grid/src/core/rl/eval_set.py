# src/core/rl/eval_set.py
"""A version-controlled copy of one dataset split, so a published score can be re-measured.

Run:
    uv run python -m src.core.rl.eval_set export --dataset seed20300000_n20000_456 --split test
    uv run python -m src.core.rl.eval_set check rl_eval_sets/seed20300000_n20000_456_test

`datasets/` is not in version control, and re-running the generator does not give the same
puzzles back: `generate_puzzle` abandons its backtracking on wall-clock time, so the dataset
is the unit of identity, not the command (`generate_dataset_v2` docstring). A best-of-N
number is therefore only reproducible on the exact puzzles it was measured on -- in the same
order, because the scorer draws one rng stream across them. This module freezes a split into
the repo and proves the frozen copy is the same content.

The layout mirrors a dataset directory: `manifest.json` next to the data. The data is JSON
Lines instead of a pickle, because a pickle runs code when it is loaded and this file is meant
for strangers. A `Puzzle` is stored as what the generator builds it from -- the layout strings
and the wall set -- so loading goes through the one parser, `parse_puzzle_layout`.

Identity is checked with the source manifest's own `content_sha256` for the split, recomputed
by `generate_dataset_v2.split_digest` over the loaded samples. That check needs nothing but
this directory, so anyone can run it.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.core.rl.baselines import DATASET_ROOT, load_split
from src.core.rl.generate_dataset_v2 import split_digest
from src.core.rl.rl_env_v2 import PuzzleSample
from src.core.utils import parse_puzzle_layout

EVAL_SET_ROOT = Path(__file__).resolve().parents[3] / "rl_eval_sets"
MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.jsonl"
SOURCE_PICKLE_FILENAME = "dataset.pkl"
FORMAT_VERSION = 1
HASH_CHUNK_BYTES = 1 << 20

Cell = tuple[int, int]


def sample_to_record(sample: PuzzleSample) -> dict[str, Any]:
    """One sample as plain JSON types. Wall endpoints keep their stored order."""
    return {
        "puzzle_layout": sample.puzzle["puzzle_layout"],
        "walls": sorted([list(a), list(b)] for a, b in sample.puzzle["walls"]),
        "solution_path": [list(cell) for cell in sample.solution_path],
    }


def record_to_sample(record: dict[str, Any]) -> PuzzleSample:
    """Rebuilds a sample exactly as `generate_puzzle` builds one: parse, then add walls."""
    puzzle = parse_puzzle_layout(record["puzzle_layout"])
    puzzle["walls"] = {(tuple(a), tuple(b)) for a, b in record["walls"]}
    solution_path: list[Cell] = [tuple(cell) for cell in record["solution_path"]]
    return PuzzleSample(puzzle, solution_path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def export_split(dataset_dir: Path, split: str, output_dir: Path) -> dict[str, Any]:
    """Writes `split` of a pickled dataset as an eval set and returns its manifest."""
    samples = load_split(dataset_dir, split)
    source_manifest = json.loads(
        (dataset_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    recorded = source_manifest.get("content_sha256", {}).get(split)
    if recorded is None:
        raise ValueError(
            f"{dataset_dir.name} has no content digest for '{split}'; an exported copy "
            f"could never be checked against it."
        )
    if split_digest(samples) != recorded:
        raise ValueError(f"{dataset_dir.name}/{split} does not match its own manifest.")

    sizes: dict[str, int] = {}
    for sample in samples:
        size = str(sample.puzzle["grid_size"][0])
        sizes[size] = sizes.get(size, 0) + 1

    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset_dir.name,
        "split": split,
        "puzzles": len(samples),
        "puzzles_per_size": dict(sorted(sizes.items())),
        "content_sha256": recorded,
        "source_pickle_sha256": file_sha256(dataset_dir / SOURCE_PICKLE_FILENAME),
        "source_manifest": source_manifest,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(sample_to_record(s), separators=(",", ":")) for s in samples)
    (output_dir / SAMPLES_FILENAME).write_text(
        "".join(f"{line}\n" for line in lines), encoding="utf-8"
    )
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_eval_set(eval_set_dir: Path) -> tuple[dict[str, Any], list[PuzzleSample]]:
    """Loads an eval set and refuses one whose content no longer matches its manifest."""
    manifest = json.loads(
        (eval_set_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    with (eval_set_dir / SAMPLES_FILENAME).open(encoding="utf-8") as handle:
        samples = [
            record_to_sample(json.loads(line)) for line in handle if line.strip()
        ]
    if split_digest(samples) != manifest["content_sha256"]:
        raise ValueError(
            f"{eval_set_dir} does not match the content digest in its manifest."
        )
    return manifest, samples


def matches_source(eval_set_dir: Path, dataset_root: Path) -> bool:
    """Stricter than the digest: same objects, wall endpoint order included.

    `split_digest` sorts each wall's endpoints, so it cannot see a flipped pair. Nothing in
    the env depends on that order today, but "equal" should not need that caveat.
    """
    manifest, samples = load_eval_set(eval_set_dir)
    source = load_split(dataset_root / manifest["dataset"], manifest["split"])
    return samples == source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Freeze a dataset split into the repo.")
    export.add_argument("--dataset", required=True)
    export.add_argument("--split", default="test")
    export.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    export.add_argument("--output-root", type=Path, default=EVAL_SET_ROOT)

    check = commands.add_parser("check", help="Verify an eval set's content digest.")
    check.add_argument("eval_set", type=Path)
    check.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Also compare object by object with the source pickle, if you have it.",
    )
    args = parser.parse_args()

    if args.command == "export":
        output_dir = args.output_root / f"{args.dataset}_{args.split}"
        manifest = export_split(
            args.dataset_root / args.dataset, args.split, output_dir
        )
        logger.success(
            f"Exported {manifest['puzzles']} puzzles {manifest['puzzles_per_size']} "
            f"-> {output_dir} (content {manifest['content_sha256'][:12]})"
        )
        return

    manifest, samples = load_eval_set(args.eval_set)
    logger.success(
        f"{args.eval_set}: {len(samples)} puzzles match content digest "
        f"{manifest['content_sha256'][:12]}"
    )
    if args.dataset_root is not None:
        if not matches_source(args.eval_set, args.dataset_root):
            logger.error("Differs from the source pickle object by object.")
            raise SystemExit(1)
        logger.success("Identical to the source pickle, object by object.")


if __name__ == "__main__":
    main()
