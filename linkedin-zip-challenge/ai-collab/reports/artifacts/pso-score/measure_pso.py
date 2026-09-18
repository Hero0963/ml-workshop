# ai-collab/reports/artifacts/pso-score/measure_pso.py
"""What PSO scores on the RL held-out boards, 4x4 and 6x6.

Two numbers per size, measured separately because only one of them depends on the clock:

- `raw`: one call at the default settings, seeded per board. Deterministic, so it is valid
  even while other work is using the machine.
- `served`: the wrapper the API served PSO through (`registry._until_verified`, rerun until
  an answer verifies, 5s of wall clock). This is the number comparable to RL's best-of-32,
  and it is only valid on an idle machine -- check `.agent-heavy-job` before running it.

Boards are the test split of an RL dataset under `datasets/rl_datasets_v2/`, read in place.
Run from the project root:
`uv run python ai-collab/reports/artifacts/pso-score/measure_pso.py raw|served <dataset_dir>`
"""

import json
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from loguru import logger

from src.core.solvers import registry
from src.core.solvers.particle_swarm_optimization import solve_puzzle_pso
from src.core.solvers.verify import is_solution

SEED = 20260919
SIZES = (4, 6)
SPLIT = "test"
SERVED_SAMPLE_PER_SIZE = 200
#: A third of the 24 logical cores. The served budget is wall clock, so oversubscribing
#: would lower the score by slowing every run down, not by making PSO worse.
WORKERS = 8
OUTPUT_DIR = Path(__file__).parent
PSO_MODULE = solve_puzzle_pso.__module__


def _load_boards(dataset_dir: Path) -> dict[int, list]:
    with (dataset_dir / "dataset.pkl").open("rb") as handle:
        dataset = pickle.load(handle)
    by_size: dict[int, list] = defaultdict(list)
    for sample in dataset["splits"][SPLIT]:
        size = sample.puzzle["grid_size"][0]
        if size in SIZES:
            by_size[size].append(sample.puzzle)
    return by_size


def _wall_count(puzzle: dict) -> int:
    return len(puzzle.get("walls", ()))


def _summarise(rows: list[dict]) -> dict:
    by_walls: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        by_walls[row["walls"]].append(row["solved"])
    return {
        "boards": len(rows),
        "solved": sum(row["solved"] for row in rows),
        "solve_rate": round(sum(row["solved"] for row in rows) / len(rows), 4),
        "mean_seconds": round(sum(row["seconds"] for row in rows) / len(rows), 4),
        "solve_rate_by_wall_count": {
            walls: round(sum(solved) / len(solved), 4)
            for walls, solved in sorted(by_walls.items())
        },
    }


def _measure_raw(by_size: dict[int, list]) -> dict:
    logger.disable(PSO_MODULE)
    result = {}
    for size in SIZES:
        rows = []
        for index, puzzle in enumerate(by_size[size]):
            random.seed(SEED + index)
            started = time.perf_counter()
            path = solve_puzzle_pso(puzzle)
            rows.append(
                {
                    "index": index,
                    "walls": _wall_count(puzzle),
                    "seconds": time.perf_counter() - started,
                    "solved": is_solution(puzzle, path),
                }
            )
        result[str(size)] = {
            **_summarise(rows),
            "solved_indices": [row["index"] for row in rows if row["solved"]],
        }
        logger.info(f"raw {size}x{size}: {result[str(size)]['solve_rate']:.4f}")
    logger.enable(PSO_MODULE)
    return result


_captured: list[str] = []


def _init_worker() -> None:
    logger.remove()
    logger.add(lambda message: _captured.append(str(message)), level="INFO")


def _serve_one(task: tuple[int, int, dict]) -> dict:
    size, index, puzzle = task
    random.seed(SEED + index)
    _captured.clear()
    served = registry._until_verified(solve_puzzle_pso)
    started = time.perf_counter()
    path = served(puzzle)
    seconds = time.perf_counter() - started
    runs = next(
        (
            int(line.split(" run ")[1].split(" ")[0])
            if " solved on run " in line
            else int(line.split("after ")[1].split(" ")[0])
            for line in _captured
            if " solved on run " in line or " gave up after " in line
        ),
        None,
    )
    return {
        "size": size,
        "index": index,
        "walls": _wall_count(puzzle),
        "seconds": round(seconds, 3),
        "runs": runs,
        "solved": is_solution(puzzle, path),
    }


def _measure_served(by_size: dict[int, list]) -> dict:
    sampler = random.Random(SEED)
    tasks = []
    for size in SIZES:
        indices = sorted(
            sampler.sample(range(len(by_size[size])), SERVED_SAMPLE_PER_SIZE)
        )
        tasks += [(size, index, by_size[size][index]) for index in indices]

    with Pool(processes=WORKERS, initializer=_init_worker) as pool:
        rows = pool.map(_serve_one, tasks, chunksize=1)

    result = {}
    for size in SIZES:
        sized = [row for row in rows if row["size"] == size]
        solved_runs = sorted(row["runs"] for row in sized if row["solved"])
        result[str(size)] = {
            **_summarise(sized),
            "sampled_indices": [row["index"] for row in sized],
            "median_runs_when_solved": (
                solved_runs[len(solved_runs) // 2] if solved_runs else None
            ),
            "rows": sized,
        }
        logger.info(f"served {size}x{size}: {result[str(size)]['solve_rate']:.4f}")
    return result


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in {"raw", "served"}:
        sys.exit(__doc__)
    mode, dataset_dir = sys.argv[1], Path(sys.argv[2])
    by_size = _load_boards(dataset_dir)

    started = time.perf_counter()
    if mode == "raw":
        measured = _measure_raw(by_size)
    else:
        measured = _measure_served(by_size)

    payload = {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "dataset": dataset_dir.name,
        "split": SPLIT,
        "seed": SEED,
        "boards_per_size": {str(size): len(by_size[size]) for size in SIZES},
        "budget_seconds": registry.HEURISTIC_TIME_BUDGET_SECONDS
        if mode == "served"
        else None,
        "workers": WORKERS if mode == "served" else 1,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "per_size": measured,
    }
    output = OUTPUT_DIR / f"pso-{mode}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote {output}")


if __name__ == "__main__":
    main()
