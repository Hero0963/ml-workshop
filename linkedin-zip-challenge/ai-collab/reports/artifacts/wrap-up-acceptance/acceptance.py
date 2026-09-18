# ai-collab/reports/artifacts/wrap-up-acceptance/acceptance.py
"""End-to-end acceptance of a running stack: every served solver, over HTTP.

The service is only exercised through its API, the way a user or the UIs reach it.
`src` is imported for two things only: the boards to send (the test fixtures plus fresh
ones from the generator, so "make a puzzle" is exercised too) and `verify.is_solution`,
the independent judge. A `200` is not taken as success -- a solver that gives up also
answers `200` -- so every returned path is judged.

Run it wherever `src` imports and the API is reachable, e.g. inside the app container:

    docker cp acceptance.py <app container>:/tmp/acceptance.py
    docker exec -w /app -e PYTHONPATH=/app <app container> \
        uv run python /tmp/acceptance.py --out /tmp/result.json
"""

import argparse
import ast
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.solvers.verify import is_solution
from src.core.tests.conftest import (
    puzzle_01_data,
    puzzle_01_layout,
    puzzle_04_data,
    puzzle_04_layout,
)
from src.core.utils import Puzzle

DEFAULT_BASE_URL = "http://127.0.0.1:7440"
GENERATOR_SEED = 20260919
GENERATED_SIZES = (4, 5, 6)
GENERATOR_TIMEOUT_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 120
PAGES = ("/api/echo/health", "/ui/", "/svelte-ui/", "/docs", "/openapi.json")


def _layout_of(puzzle: Puzzle) -> list[list[str]]:
    """The two-character layout `parse_puzzle_layout` reads, rebuilt from a puzzle."""
    blocked = puzzle.get("blocked_cells", set())
    return [
        [
            "xx" if (r, c) in blocked else (f"{value:02d}" if value else "  ")
            for c, value in enumerate(row)
        ]
        for r, row in enumerate(puzzle["grid"])
    ]


def _boards() -> list[tuple[str, Puzzle, list[list[str]]]]:
    boards = [
        ("puzzle_01 (6x6, fixture)", puzzle_01_data, puzzle_01_layout),
        ("puzzle_04 (7x7, fixture)", puzzle_04_data, puzzle_04_layout),
    ]
    logger.disable("src.core.puzzle_generation")
    for size in GENERATED_SIZES:
        random.seed(GENERATOR_SEED + size)
        generated = None
        while generated is None:
            generated = generate_puzzle(
                m=size,
                n=size,
                has_walls=True,
                timeout_per_attempt=GENERATOR_TIMEOUT_SECONDS,
            )
        puzzle, _solution = generated
        boards.append(
            (
                f"generated {size}x{size}, {len(puzzle['walls'])} walls",
                puzzle,
                _layout_of(puzzle),
            )
        )
    logger.enable("src.core.puzzle_generation")
    return boards


def _get(base_url: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(
            base_url + path, timeout=REQUEST_TIMEOUT_SECONDS
        ) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _solve(
    base_url: str, solver: str, layout: list, walls: set
) -> tuple[int, dict, float]:
    body = json.dumps(
        {
            "puzzle_layout_str": repr(layout),
            "walls_str": repr(walls) if walls else "set()",
            "solver_name": solver,
        }
    ).encode()
    request = urllib.request.Request(
        base_url + "/api/solver/solve",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as r:
            status, payload = r.status, json.loads(r.read())
    except urllib.error.HTTPError as error:
        status, payload = error.code, json.loads(error.read() or b"{}")
    return status, payload, time.perf_counter() - started


def _path_of(solution_path: str) -> list[tuple[int, int]] | None:
    if "could not find" in solution_path:
        return None
    return [ast.literal_eval(step) for step in solution_path.split(" -> ")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pages = {path: _get(args.base_url, path)[0] for path in PAGES}
    status, raw = _get(args.base_url, "/api/solver/list")
    solvers = [entry["name"] for entry in json.loads(raw)] if status == 200 else []

    rows = []
    for board_name, puzzle, layout in _boards():
        for solver in solvers:
            status, payload, seconds = _solve(
                args.base_url, solver, layout, puzzle.get("walls", set())
            )
            path = _path_of(payload.get("solution_path", "")) if status == 200 else None
            rows.append(
                {
                    "board": board_name,
                    "solver": solver,
                    "status": status,
                    "seconds": round(seconds, 3),
                    "solved": is_solution(puzzle, path),
                    "gave_up": status == 200 and path is None,
                    "has_images": bool(payload.get("solution_final_image_b64")),
                    "detail": payload.get("detail"),
                }
            )

    result = {
        "base_url": args.base_url,
        "pages": pages,
        "solvers": solvers,
        "rows": rows,
    }
    for path, code in pages.items():
        print(f"{code}  {path}")
    print(f"{len(solvers)} solvers listed: {solvers}")
    for row in rows:
        verdict = "SOLVED" if row["solved"] else ("gave up" if row["gave_up"] else "-")
        detail = f"  {row['detail'][:70]}" if row["detail"] else ""
        print(
            f"{row['status']}  {verdict:8} {row['seconds']:7.3f}s  "
            f"{row['solver']:24} {row['board']}{detail}"
        )
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
