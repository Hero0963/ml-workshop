# src/core/vl_models/score_predictions.py
"""Scores held-out predictions produced elsewhere (Colab) against their labels.

P4c fine-tunes on Colab, where this repository is not installed. Rather than
reimplement the metrics in the notebook -- which is how the P1 numbers and the
shipped parser drifted apart in the first place -- the notebook only writes raw
model output to a JSONL file and this module scores it locally, through the same
code the API runs:

*   ``puzzle_parser.parse_model_output`` turns model text into a ``Puzzle``,
    dropping walls that are out of bounds or between non-neighbours exactly as the
    endpoint will;
*   ``benchmark.score_layout`` / ``score_walls`` compute the metric layers the
    published baseline was measured with.

Input is one JSON object per line with at least ``label`` (the ground-truth JSON
string from ``metadata.jsonl``) and ``raw_output`` (what the model generated).
Any other keys -- ``file_name``, ``wall_count``, ``generation_seconds`` -- are
carried through to the per-item results.

Three layers of verdict are reported, and they are not the same question:

*   ``exact_match`` -- the parse is byte-perfect;
*   ``solution_valid_on_truth`` -- **the product metric**: solve the board the
    model read, then check that answer against the board that was really there.
    Weaker than ``exact_match`` on purpose, because misreading a wall the route
    never touches costs the user nothing;
*   ``solvable`` -- a check that needs no ground truth at all, because the
    generator builds every board from a Hamiltonian path and so every real board
    has a solution. An unsolvable prediction is therefore *known* to be a misread,
    which makes it shippable as a confidence signal. The dangerous case is the
    opposite one, counted as ``solvable_but_wrong``: the solver happily returns a
    route for a board that was not in the picture, and nothing complains.

``exact_match`` is the strictest number, not wall F1. A puzzle is only usable when
*every* wall is right: at six walls per board, a per-wall accuracy of 0.85 yields
38% of boards fully correct, so an F1 around 0.85 is a much weaker result than it
sounds. The per-wall-count breakdown is there to show where that falls apart.

Usage:
    uv run python -m src.core.vl_models.score_predictions predictions.jsonl
    uv run python -m src.core.vl_models.score_predictions predictions.jsonl --no-solve
"""

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.core.solvers.cp import solve_puzzle_cp
from src.core.utils import Puzzle
from src.core.vl_models.benchmark import normalize_layout, score_layout, score_walls
from src.core.vl_models.puzzle_parser import (
    ModelOutputError,
    parse_model_output,
    to_puzzle,
)
from src.core.vl_models.schema import SimplePuzzleOutput

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ai-collab" / "reports" / "artifacts" / "vl-p4c"

LABEL_FIELD = "label"
RAW_OUTPUT_FIELD = "raw_output"
CARRIED_FIELDS = ("file_name", "generation_seconds")

# Guard against feeding CP-SAT a hallucinated 40x40 board and waiting forever.
SOLVE_MAX_CELLS = 100

# Zip paths start on waypoint 1 and collect the rest in ascending order.
FIRST_WAYPOINT = 1


class LabelError(ValueError):
    """The ground-truth label itself is unusable, which is a dataset bug."""


def truth_from_label(label: str) -> Puzzle:
    """Reads one ``metadata.jsonl`` label into a solver-ready ``Puzzle``.

    The label goes through the same conversion the prediction does, so a difference
    between the two can only come from the model, never from asymmetric handling.
    """
    try:
        validated = SimplePuzzleOutput(**json.loads(label))
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise LabelError(f"Label is not a valid puzzle: {error}") from error
    return to_puzzle(validated).puzzle


def path_is_legal(puzzle: Puzzle, path: list[tuple[int, int]]) -> bool:
    """Would this path be accepted as a solution to ``puzzle``?

    The rules mirror ``dfs.py``'s ``_backtrack``: single steps between orthogonal
    neighbours, no cell twice, no blocked cell, no crossing a wall, every visitable
    cell covered, and numbered cells entered in ascending order.

    This is what makes the pipeline's real success criterion measurable. Solving the
    *predicted* board and checking the answer against the *true* board is a weaker
    test than demanding the two boards be identical -- misreading a wall the solution
    never touches costs the user nothing -- and it is the one that matches what the
    product does.
    """
    height, width = puzzle["grid_size"]
    grid, blocked, walls = puzzle["grid"], puzzle["blocked_cells"], puzzle["walls"]
    num_map = puzzle["num_map"]

    if not path or (num_map and path[0] != num_map.get(FIRST_WAYPOINT)):
        return False
    if len(set(path)) != len(path):
        return False
    if len(path) != height * width - len(blocked):
        return False

    expected = FIRST_WAYPOINT
    previous: tuple[int, int] | None = None
    for cell in path:
        row, col = cell
        if not (0 <= row < height and 0 <= col < width) or cell in blocked:
            return False
        if previous is not None:
            if abs(row - previous[0]) + abs(col - previous[1]) != 1:
                return False
            if tuple(sorted((previous, cell))) in walls:
                return False
        number = grid[row][col]
        if number > 0:
            if number != expected:
                return False
            expected += 1
        previous = cell

    return not num_map or expected > max(num_map)


def score_record(record: dict[str, Any], solve: bool = True) -> dict[str, Any]:
    """Scores one prediction. Never raises on model output, only on a bad label."""
    truth = truth_from_label(record[LABEL_FIELD])
    truth_layout, truth_walls = truth["puzzle_layout"], truth["walls"]
    result: dict[str, Any] = {
        key: record[key] for key in CARRIED_FIELDS if key in record
    }
    result["truth_wall_count"] = len(truth_walls)

    try:
        parsed = parse_model_output(record[RAW_OUTPUT_FIELD])
    except ModelOutputError as error:
        result["json_parsed"] = False
        result["parse_error"] = str(error)
        result["exact_match"] = False
        result["solution_valid_on_truth"] = False
        return result

    predicted_layout = parsed.puzzle["puzzle_layout"]
    predicted_walls = parsed.puzzle["walls"]

    result["json_parsed"] = True
    result["parser_warnings"] = list(parsed.warnings)
    result["layout"] = score_layout(predicted_layout, truth_layout)
    result["walls"] = score_walls(predicted_walls, truth_walls)
    result["exact_match"] = (
        normalize_layout(predicted_layout) == normalize_layout(truth_layout)
        and predicted_walls == truth_walls
    )
    if solve:
        path = _solve(parsed.puzzle)
        result["solvable"] = path is not None
        # An unsolvable prediction is the failure the user can be shown; a solvable
        # one that is wrong on the true board is the failure nobody notices.
        result["solution_valid_on_truth"] = path is not None and path_is_legal(
            truth, path
        )
    return result


def _solve(puzzle: Puzzle) -> list[tuple[int, int]] | None:
    height, width = puzzle["grid_size"]
    if height * width > SOLVE_MAX_CELLS:
        return None
    try:
        return solve_puzzle_cp(puzzle)
    # A malformed prediction can break assumptions solve_puzzle_cp does not check.
    except Exception as error:
        logger.warning(f"CP-SAT raised on a predicted puzzle: {error}")
        return None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates per-item results, plus a breakdown by how many walls are on the board."""
    total = len(results)
    parsed = [result for result in results if result["json_parsed"]]
    walled = [result for result in parsed if result["truth_wall_count"] > 0]

    true_positives = sum(result["walls"]["true_positives"] for result in parsed)
    predicted_total = sum(result["walls"]["predicted_count"] for result in parsed)
    truth_total = sum(result["walls"]["truth_count"] for result in parsed)

    return {
        "items": total,
        "json_parse_rate": _rate(len(parsed), total),
        "shape_correct": sum(1 for r in parsed if r["layout"]["shape_match"]),
        "mean_cell_accuracy": _mean([r["layout"]["cell_accuracy"] for r in parsed]),
        "mean_waypoint_recall": _mean([r["layout"]["waypoint_recall"] for r in parsed]),
        "mean_wall_f1": _mean([r["walls"]["f1"] for r in parsed]),
        # Wall-free boards score a free 1.0; the walled subset is the honest number.
        "mean_wall_f1_walled_only": _mean([r["walls"]["f1"] for r in walled]),
        "walled_items": len(walled),
        # Micro rates pool every wall in the set, so they are far less noisy than an
        # average of per-board F1 and are what the per-wall accuracy target refers to.
        "micro_wall_precision": _rate(true_positives, predicted_total),
        "micro_wall_recall": _rate(true_positives, truth_total),
        "exact_match": sum(1 for r in results if r["exact_match"]),
        "exact_match_rate": _rate(sum(1 for r in results if r["exact_match"]), total),
        # The product metric: the solver was run on what the model read, and the answer
        # it produced is legal on the board that was actually in the picture.
        "solution_valid_on_truth": sum(
            1 for r in results if r.get("solution_valid_on_truth")
        ),
        "solution_valid_rate": _rate(
            sum(1 for r in results if r.get("solution_valid_on_truth")), total
        ),
        "solvable": sum(1 for r in parsed if r.get("solvable")),
        # Solvable but wrong is the silent failure: the user is handed a route that
        # walks through a wall that was there in the image.
        "solvable_but_wrong": sum(
            1
            for r in parsed
            if r.get("solvable") and not r.get("solution_valid_on_truth")
        ),
        "by_wall_count": _by_wall_count(results),
    }


def _by_wall_count(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        buckets.setdefault(result["truth_wall_count"], []).append(result)
    return {
        str(wall_count): {
            "items": len(bucket),
            "exact_match_rate": _rate(
                sum(1 for r in bucket if r["exact_match"]), len(bucket)
            ),
            "solution_valid_rate": _rate(
                sum(1 for r in bucket if r.get("solution_valid_on_truth")), len(bucket)
            ),
            "mean_wall_f1": _mean(
                [r["walls"]["f1"] for r in bucket if r["json_parsed"]]
            ),
        }
        for wall_count, bucket in sorted(buckets.items())
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def read_predictions(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def score_predictions(
    records: list[dict[str, Any]], solve: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = [score_record(record, solve=solve) for record in records]
    return results, summarize(results)


def format_summary(summary: dict[str, Any]) -> str:
    """A compact table, because the JSON artifact is for machines and this is for me."""
    lines = [
        f"items                    {summary['items']}",
        f"JSON parse rate          {summary['json_parse_rate']:.3f}",
        f"grid size correct        {summary['shape_correct']}/{summary['items']}",
        f"mean cell accuracy       {summary['mean_cell_accuracy']:.3f}",
        f"mean waypoint recall     {summary['mean_waypoint_recall']:.3f}",
        f"mean wall F1 (walled)    {summary['mean_wall_f1_walled_only']:.3f}"
        f"  over {summary['walled_items']} boards",
        f"micro wall precision     {summary['micro_wall_precision']:.3f}",
        f"micro wall recall        {summary['micro_wall_recall']:.3f}",
        "",
        f"EXACT MATCH              {summary['exact_match']}/{summary['items']}"
        f"  ({summary['exact_match_rate']:.3f})   the parse is perfect",
        f"SOLUTION VALID           {summary['solution_valid_on_truth']}/{summary['items']}"
        f"  ({summary['solution_valid_rate']:.3f})   the pipeline's answer is right",
        f"predicted board solvable {summary['solvable']}"
        f"   (of which wrong: {summary['solvable_but_wrong']}  <- silent failures)",
        "",
        f"{'walls':>7}{'n':>6}{'exact':>8}{'valid':>8}{'wall F1':>9}",
    ]
    for wall_count, bucket in summary["by_wall_count"].items():
        lines.append(
            f"{wall_count:>7}{bucket['items']:>6}"
            f"{bucket['exact_match_rate']:>8.2f}{bucket['solution_valid_rate']:>8.2f}"
            f"{bucket['mean_wall_f1']:>9.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "predictions", type=Path, help="JSONL with 'label' and 'raw_output' per line"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-solve",
        dest="solve",
        action="store_false",
        help="skip the CP-SAT solvability check on each predicted board",
    )
    args = parser.parse_args()

    records = read_predictions(args.predictions)
    logger.info(f"scoring {len(records)} predictions from {args.predictions}")
    results, summary = score_predictions(records, solve=args.solve)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.predictions.stem}_scored.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    logger.info("\n" + format_summary(summary))
    logger.info(f"artifact written to {out_path}")


if __name__ == "__main__":
    main()
