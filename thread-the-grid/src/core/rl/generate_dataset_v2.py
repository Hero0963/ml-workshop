# src/core/rl/generate_dataset_v2.py
"""Builds a reproducible puzzle dataset for the one-stroke RL environment.

Run:  uv run python -m src.core.rl.generate_dataset_v2 --count 300 --sizes 4,5,6,7

Why not reuse `generate_rl_dataset.py`: that script drops the ground-truth path
(`generate_rl_dataset.py:59` unpacks `puzzle, _ = result`) and has no seed control.
Reverse curriculum needs the path, and train/val/test splits need determinism, so this
is a separate script and the old one is left untouched as the v1 control.

Every puzzle is generated from `base_seed + index` and each worker seeds itself, so pool
scheduling does not affect the result. Generation can fail by parity on odd open grids
(see `ai-collab/reports/2026-08-15_a0-env-v1-findings.md` §6), so each task retries with
derived seeds and the manifest records how often that happened.

⚠ **Re-running the same command does not reproduce a dataset bit-for-bit.** `generate_puzzle`
abandons its randomized backtracking on *wall-clock* time, so whether a given attempt is cut
short depends on machine load, and a clipped attempt is retried under a different derived
seed. The VLM track measured the same effect on the shared generator: two runs of one seed
differed on 8 of 30 samples at a 0.5s budget. **So the dataset is the unit of identity, not
the command** — the manifest carries a SHA-256 over the canonical content of each split, and
`--verify <name>` recomputes them. Build once, and refer to a dataset by its digest.
"""

import argparse
import hashlib
import json
import os
import pickle
import random
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
from tqdm import tqdm

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.rl_env_v2 import PuzzleSample
from src.core.rl.train_config import capped_worker_count

OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "rl_datasets_v2"
DEFAULT_SIZES = (4, 5, 6, 7)
DEFAULT_COUNT_PER_SIZE = 300
DEFAULT_BASE_SEED = 20260815
DEFAULT_SPLIT = (0.8, 0.1, 0.1)
MAX_GENERATION_RETRIES = 8
# The generator's default is 20s, but a 7x7 search that is going to succeed finishes in
# under 0.5s (measured 2026-08-15: successes max 0.415s at a 0.5s cutoff, 1.606s at 2s).
# Long waits are spent proving that a wrong-parity start cell is impossible, so a short
# cutoff plus a retry is ~18x cheaper than waiting out the default.
DEFAULT_TIMEOUT_PER_ATTEMPT = 0.5
WALL_PROBABILITY = 0.5


def _generate_one(task: dict[str, Any]) -> dict[str, Any] | None:
    """Generates a single puzzle deterministically from `task['seed']`."""
    # Every worker is a fresh process under `spawn`, so this has to happen *here*: a call
    # in the parent never reaches the child. `generate_puzzle` logs a line per attempt, and
    # a 40k build emitted ~90k lines from 16 processes onto one inherited stderr. When that
    # stderr is a pipe the writers deadlock -- measured 2026-08-29, an identical build hangs
    # indefinitely piped and finishes in seconds either with this line or writing to a file.
    # `vl_models/dataset_builder.py:291` disables the same logger in its worker for the same
    # reason, and its docstring records the same hang.
    logger.disable("src.core.puzzle_generation.puzzle_generator")

    for attempt in range(MAX_GENERATION_RETRIES):
        seed = task["seed"] * 100 + attempt
        random.seed(seed)
        has_walls = random.random() < WALL_PROBABILITY
        result = generate_puzzle(
            m=task["size"],
            n=task["size"],
            has_walls=has_walls,
            num_blocked_cells=0,
            timeout_per_attempt=task["timeout"],
        )
        if result is not None:
            puzzle, solution_path = result
            return {
                "sample": PuzzleSample(puzzle, solution_path),
                "size": task["size"],
                "seed": seed,
                "retries": attempt,
                "has_walls": has_walls,
            }
    logger.error(f"Task {task['index']} (size {task['size']}) failed every retry.")
    return None


def sample_fingerprint(sample: PuzzleSample) -> str:
    """Canonical text for one sample, independent of how Python happens to serialise it.

    Hashing the pickle would hash the *encoding* -- set iteration order, protocol version --
    rather than the puzzle, so two datasets with identical content could disagree.
    """
    puzzle = sample.puzzle
    return json.dumps(
        {
            "grid_size": list(puzzle["grid_size"]),
            "grid": puzzle["grid"],
            "walls": sorted(sorted(map(list, edge)) for edge in puzzle["walls"]),
            "blocked_cells": sorted(map(list, puzzle["blocked_cells"])),
            "num_map": {
                str(number): list(cell)
                for number, cell in sorted(puzzle["num_map"].items())
            },
            "solution_path": [list(cell) for cell in sample.solution_path],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def split_digest(samples: Sequence[PuzzleSample]) -> str:
    """SHA-256 over the split's samples in order; order is part of the identity."""
    digest = hashlib.sha256()
    for sample in samples:
        payload = sample_fingerprint(sample).encode("utf-8")
        digest.update(f"{len(payload)}:".encode("utf-8"))
        digest.update(payload)
    return digest.hexdigest()


def content_digests(splits: dict[str, list[PuzzleSample]]) -> dict[str, str]:
    return {name: split_digest(samples) for name, samples in sorted(splits.items())}


def verify_dataset(dataset_dir: Path) -> bool:
    """Recomputes every split digest and checks it against the manifest."""
    with (dataset_dir / "dataset.pkl").open("rb") as handle:
        dataset = pickle.load(handle)
    recorded = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    ).get("content_sha256")
    if not recorded:
        logger.error(
            f"{dataset_dir.name} predates content digests; rebuild to get them."
        )
        return False

    recomputed = content_digests(dataset["splits"])
    ok = True
    for name, digest in sorted(recomputed.items()):
        matched = recorded.get(name) == digest
        ok = ok and matched
        logger.info(f"  {name:5s} {digest[:16]}... {'ok' if matched else 'MISMATCH'}")
    return ok


def deduplicate(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drops repeat puzzles, keeping the first by seed order.

    Small boards run out of distinct puzzles long before a large request is filled: at
    20,000 requested, 4x4 yielded 97.2% unique and put 111 puzzles in both train and test
    (measured 2026-08-29), which would flatter any held-out score. Deduplicating before the
    split makes the three splits disjoint by construction rather than by luck.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        fingerprint = sample_fingerprint(record["sample"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(record)
    return unique, len(records) - len(unique)


def _split_indices(
    total: int, split: tuple[float, float, float]
) -> dict[str, tuple[int, int]]:
    train_end = int(total * split[0])
    val_end = train_end + int(total * split[1])
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total),
    }


def build_dataset(
    sizes: tuple[int, ...],
    count_per_size: int,
    base_seed: int,
    split: tuple[float, float, float],
    processes: int | None,
    timeout_per_attempt: float = DEFAULT_TIMEOUT_PER_ATTEMPT,
) -> dict[str, Any]:
    """Generates every puzzle, then slices each size independently into the splits."""
    tasks: list[dict[str, Any]] = []
    for size in sizes:
        for _ in range(count_per_size):
            index = len(tasks)
            tasks.append(
                {
                    "index": index,
                    "size": size,
                    "seed": base_seed + index,
                    "timeout": timeout_per_attempt,
                }
            )

    workers = processes or capped_worker_count()
    logger.info(
        f"Generating {len(tasks)} puzzles across sizes {sizes} on {workers} workers "
        f"(of {os.cpu_count()} logical cores)..."
    )
    results: list[dict[str, Any]] = []
    with Pool(processes=workers) as pool, tqdm(total=len(tasks)) as progress:
        for result in pool.imap_unordered(_generate_one, tasks):
            if result is not None:
                results.append(result)
            progress.update(1)

    splits: dict[str, list[PuzzleSample]] = {"train": [], "val": [], "test": []}
    per_size_stats: dict[int, dict[str, Any]] = {}
    for size in sizes:
        of_size = sorted(
            (r for r in results if r["size"] == size), key=lambda r: r["seed"]
        )
        of_size, duplicates_dropped = deduplicate(of_size)
        bounds = _split_indices(len(of_size), split)
        for name, (start, end) in bounds.items():
            splits[name].extend(r["sample"] for r in of_size[start:end])
        per_size_stats[size] = {
            "generated": len(of_size),
            "requested": count_per_size,
            "duplicates_dropped": duplicates_dropped,
            "retried": sum(1 for r in of_size if r["retries"] > 0),
            "walls": sum(1 for r in of_size if r["has_walls"]),
            "split_sizes": {name: end - start for name, (start, end) in bounds.items()},
        }

    return {
        "manifest": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sizes": list(sizes),
            "count_per_size": count_per_size,
            "base_seed": base_seed,
            "split": list(split),
            "timeout_per_attempt": timeout_per_attempt,
            "total_generated": len(results),
            "total_requested": len(tasks),
            "per_size": per_size_stats,
            "content_sha256": content_digests(splits),
        },
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT_PER_SIZE)
    parser.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker processes; defaults to a capped share of the cores, not all of them.",
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_PER_ATTEMPT,
        help="Per-attempt search cutoff in seconds; short is faster (see module docstring).",
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        help="Recheck an existing dataset directory against its manifest digests, then exit.",
    )
    args = parser.parse_args()

    if args.verify:
        dataset_dir = OUTPUT_ROOT / args.verify
        logger.info(f"Verifying {dataset_dir}")
        raise SystemExit(0 if verify_dataset(dataset_dir) else 1)

    sizes = tuple(int(size) for size in args.sizes.split(","))
    dataset = build_dataset(
        sizes=sizes,
        count_per_size=args.count,
        base_seed=args.base_seed,
        split=DEFAULT_SPLIT,
        processes=args.processes,
        timeout_per_attempt=args.timeout,
    )

    name = (
        args.name or f"seed{args.base_seed}_n{args.count}_{'-'.join(map(str, sizes))}"
    )
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "dataset.pkl").open("wb") as handle:
        pickle.dump(dataset, handle)
    (output_dir / "manifest.json").write_text(
        json.dumps(dataset["manifest"], indent=2), encoding="utf-8"
    )

    manifest = dataset["manifest"]
    logger.success(
        f"Generated {manifest['total_generated']}/{manifest['total_requested']} puzzles -> {output_dir}"
    )
    for size, stats in manifest["per_size"].items():
        logger.info(f"  size {size}: {stats}")
    for name, digest in manifest["content_sha256"].items():
        logger.info(f"  {name:5s} sha256 {digest}")


if __name__ == "__main__":
    main()
