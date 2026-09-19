# ai-collab/reports/artifacts/heuristic-api-budget/measure_budget.py
"""Measures what the six heuristics actually return, raw and behind `_until_verified`.

Answers the two questions the registry's design rests on: how often one run at the
default settings is a real solution, and what a fixed wall-clock budget buys on top.
Run from the project root: `uv run python ai-collab/reports/artifacts/heuristic-api-budget/measure_budget.py`
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.core.solvers import registry
from src.core.solvers.verify import is_solution
from src.core.tests.conftest import puzzles_to_test

SEED = 20260912
REPEATS_PER_PUZZLE = 5
OUTPUT_DIR = Path(__file__).parent

HEURISTICS = [
    entry for entry in registry.SOLVER_ENTRIES if entry.kind == registry.HEURISTIC
]


def _measure_raw() -> list[dict]:
    """One call to the bare solver, exactly as it was written, repeated per puzzle."""
    rows = []
    for entry in HEURISTICS:
        raw = entry.solve.__wrapped__
        logger.disable(raw.__module__)
        for puzzle, _expected, puzzle_id in puzzles_to_test:
            for repeat in range(REPEATS_PER_PUZZLE):
                random.seed(SEED + repeat)
                started = time.perf_counter()
                path = raw(puzzle)
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "solver": entry.name,
                        "puzzle": puzzle_id,
                        "repeat": repeat,
                        "seconds": round(elapsed, 4),
                        "returned_a_path": bool(path),
                        "is_solution": is_solution(puzzle, path),
                    }
                )
        logger.enable(raw.__module__)
    return rows


def _measure_served() -> list[dict]:
    """The wrapped solver the API actually serves, at the production budget."""
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(str(message)), level="INFO")
    rows = []
    try:
        for entry in HEURISTICS:
            for puzzle, _expected, puzzle_id in puzzles_to_test:
                random.seed(SEED)
                captured.clear()
                started = time.perf_counter()
                path = entry.solve(puzzle)
                elapsed = time.perf_counter() - started
                runs = next(
                    (
                        int(line.split(" run ")[1].split(" ")[0])
                        if " solved on run " in line
                        else int(line.split("after ")[1].split(" ")[0])
                        for line in captured
                        if " run " in line or "after " in line
                    ),
                    None,
                )
                rows.append(
                    {
                        "solver": entry.name,
                        "puzzle": puzzle_id,
                        "seconds": round(elapsed, 3),
                        "runs": runs,
                        "solved": is_solution(puzzle, path),
                        "served_a_path": bool(path),
                    }
                )
    finally:
        logger.remove(sink_id)
    return rows


def main() -> None:
    logger.info("raw single-run pass")
    raw_rows = _measure_raw()
    logger.info("served pass at the production budget")
    served_rows = _measure_served()

    payload = {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "repeats_per_puzzle": REPEATS_PER_PUZZLE,
        "budget_seconds": registry.HEURISTIC_TIME_BUDGET_SECONDS,
        "puzzles": [puzzle_id for _p, _s, puzzle_id in puzzles_to_test],
        "raw_single_runs": raw_rows,
        "served_at_budget": served_rows,
    }
    out = OUTPUT_DIR / "budget-measurements.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(f"wrote {out}")
    for entry in HEURISTICS:
        raw = [r for r in raw_rows if r["solver"] == entry.name]
        served = [r for r in served_rows if r["solver"] == entry.name]
        solved_raw = sum(r["is_solution"] for r in raw)
        logger.info(
            f"{entry.name}: raw {solved_raw}/{len(raw)} solved, "
            f"{sum(r['seconds'] for r in raw) / len(raw):.3f}s per run | "
            f"served {sum(r['solved'] for r in served)}/{len(served)}, "
            f"{sum(r['seconds'] for r in served) / len(served):.2f}s per request"
        )


if __name__ == "__main__":
    main()
