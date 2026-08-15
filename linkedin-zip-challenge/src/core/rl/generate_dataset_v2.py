# src/core/rl/generate_dataset_v2.py
"""Builds a reproducible puzzle dataset for the one-stroke RL environment.

Run:  uv run python -m src.core.rl.generate_dataset_v2 --count 300 --sizes 4,5,6,7

Why not reuse `generate_rl_dataset.py`: that script drops the ground-truth path
(`generate_rl_dataset.py:59` unpacks `puzzle, _ = result`) and has no seed control.
Reverse curriculum needs the path, and train/val/test splits need determinism, so this
is a separate script and the old one is left untouched as the v1 control.

Every puzzle is generated from `base_seed + index`, so the same arguments reproduce the
same dataset regardless of how the worker pool schedules the tasks. Generation can fail
by parity on odd open grids (see `ai-collab/reports/2026-08-15_a0-env-v1-findings.md` §6),
so each task retries with derived seeds and the manifest records how often that happened.
"""

import argparse
import json
import pickle
import random
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.rl_env_v2 import PuzzleSample

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

    logger.info(f"Generating {len(tasks)} puzzles across sizes {sizes}...")
    results: list[dict[str, Any]] = []
    with Pool(processes=processes) as pool, tqdm(total=len(tasks)) as progress:
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
        bounds = _split_indices(len(of_size), split)
        for name, (start, end) in bounds.items():
            splits[name].extend(r["sample"] for r in of_size[start:end])
        per_size_stats[size] = {
            "generated": len(of_size),
            "requested": count_per_size,
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
        },
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT_PER_SIZE)
    parser.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_PER_ATTEMPT,
        help="Per-attempt search cutoff in seconds; short is faster (see module docstring).",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
