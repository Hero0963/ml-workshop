# src/core/tests/rl/test_solver_service.py
"""Tests for serving a trained policy as a solver (stage A5).

The end-to-end test is skipped when no checkpoint is on disk, because `models/` is not in
version control -- but the *invariants* that make serving possible are tested
unconditionally, since those are what break silently:

*   a puzzle has no solution when you are trying to solve it, so `inference_sample` has to
    be able to build the env from the puzzle alone;
*   a missing checkpoint and an unsupported board are different answers to the caller, and
    neither is a crash.
"""

import random

import pytest

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl import solver_service
from src.core.rl.baselines import make_eval_env
from src.core.rl.solver_service import (
    FIRST_WAYPOINT_NUMBER,
    RUN_ID_BY_SIZE,
    ModelUnavailableError,
    UnsupportedBoardError,
    checkpoint_path,
    inference_sample,
    solve_puzzle_rl,
    supported_sizes,
)
from src.core.utils import Puzzle

GENERATOR_SEED = 20260911
SERVED_SIZE = 4


@pytest.fixture(scope="module")
def puzzle() -> Puzzle:
    random.seed(GENERATOR_SEED)
    result = generate_puzzle(
        m=SERVED_SIZE, n=SERVED_SIZE, has_walls=True, timeout_per_attempt=5.0
    )
    assert result is not None, "generator returned None; retry with another seed"
    generated, _ = result
    return generated


def test_the_env_only_needs_the_start_cell_not_the_solution(puzzle: Puzzle) -> None:
    """Pins why serving without a solution is sound, not a stub.

    `PuzzleSample.solution_path` exists for training. With `reverse_curriculum_k=None`
    the env reads exactly one element of it -- the starting cell -- which the puzzle
    already states as waypoint 1. If the env ever reads further, this fails loudly here
    rather than producing a solver that quietly starts in the wrong place.
    """
    sample = inference_sample(puzzle)
    assert sample.solution_path == [puzzle["num_map"][FIRST_WAYPOINT_NUMBER]]

    env = make_eval_env(sample)
    _, info = env.reset(seed=0)
    assert info["agent_location"] == puzzle["num_map"][FIRST_WAYPOINT_NUMBER]
    assert info["coverage"] > 0.0


def test_a_puzzle_without_a_first_waypoint_is_rejected(puzzle: Puzzle) -> None:
    headless: Puzzle = {
        **puzzle,
        "num_map": {
            number: cell
            for number, cell in puzzle["num_map"].items()
            if number != FIRST_WAYPOINT_NUMBER
        },
    }
    with pytest.raises(UnsupportedBoardError, match="nowhere to start"):
        inference_sample(headless)


def test_an_untrained_board_size_is_a_request_error() -> None:
    unsupported = max(RUN_ID_BY_SIZE) + 1
    fake: Puzzle = {"grid_size": (unsupported, unsupported), "num_map": {1: (0, 0)}}
    with pytest.raises(UnsupportedBoardError, match="No RL policy"):
        solve_puzzle_rl(fake)


def test_a_missing_checkpoint_is_a_service_error(monkeypatch, puzzle: Puzzle) -> None:
    """Distinct from the above: the size is served, the weights just are not there."""
    monkeypatch.setitem(RUN_ID_BY_SIZE, SERVED_SIZE, "no_such_run_id")
    solver_service.load_policy.cache_clear()
    with pytest.raises(ModelUnavailableError, match="No checkpoint at"):
        solve_puzzle_rl(puzzle)
    solver_service.load_policy.cache_clear()


def test_supported_sizes_is_sorted_and_matches_the_mapping() -> None:
    assert supported_sizes() == sorted(RUN_ID_BY_SIZE)


@pytest.mark.skipif(
    not checkpoint_path(RUN_ID_BY_SIZE[SERVED_SIZE]).exists(),
    reason="no checkpoint on disk; models/ is not in version control",
)
def test_solves_a_real_board_end_to_end(puzzle: Puzzle) -> None:
    path = solve_puzzle_rl(puzzle, attempts=32)
    assert path is not None, (
        "4x4 policy failed 32 attempts; it scores ~0.95 at best-of-16"
    )

    blocked = puzzle.get("blocked_cells", set())
    expected_cells = SERVED_SIZE * SERVED_SIZE - len(blocked)
    assert path[0] == puzzle["num_map"][FIRST_WAYPOINT_NUMBER]
    assert len(path) == expected_cells, (
        "a solution covers every visitable cell exactly once"
    )
    assert len(set(path)) == len(path), "the path revisits a cell"

    # Waypoints have to appear in ascending order along the path.
    order = [
        number
        for cell in path
        for number, position in puzzle["num_map"].items()
        if position == cell
    ]
    assert order == sorted(order)
