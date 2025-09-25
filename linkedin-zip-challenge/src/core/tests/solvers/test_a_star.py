# src/core/tests/solvers/test_a_star.py

import pytest
from loguru import logger
from src.core.solvers.a_star import solve_puzzle_a_star
from src.core.tests.conftest import puzzles_to_test


# By setting ids=[p[2] for p in puzzles_to_test], we give each test a readable name.
@pytest.mark.parametrize(
    "puzzle_data, expected_solution, puzzle_id",
    puzzles_to_test,
    ids=[p[2] for p in puzzles_to_test],
)
def test_a_star_solver(puzzle_data, expected_solution, puzzle_id):
    """
    Tests the A* solver against the standard set of puzzles from conftest.py.
    """
    logger.info(f"--- [A* Solver] Starting test for: {puzzle_id} ---")
    logger.debug(f"Puzzle input: {puzzle_data}")

    actual = solve_puzzle_a_star(puzzle_data)
    logger.info(f"Solution found for {puzzle_id} via A*: {actual}")

    if expected_solution is not None:
        assert (
            actual == expected_solution
        ), f"[A* Solver] returned the wrong solution for {puzzle_id}."
    else:
        # For puzzles without a known solution, just check for completeness.
        assert (
            actual is not None
        ), f"[A* Solver] did not find a solution for {puzzle_id}."

        grid = puzzle_data["grid"]
        blocked_cells = puzzle_data.get("blocked_cells", set())
        expected_len = len(grid) * len(grid[0]) - len(blocked_cells)
        assert (
            len(actual) == expected_len
        ), f"[A* Solver] Path length for {puzzle_id} should be {expected_len}, but was {len(actual)}."

    logger.info(
        f"--- [A* Solver] Finished test for: {puzzle_id} ---\
"
    )
