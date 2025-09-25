# src/core/tests/solvers/test_cp.py

import pytest
from loguru import logger
from src.core.solvers.cp import solve_puzzle_cp
from src.core.tests.conftest import puzzles_to_test


# By setting ids=[p[2] for p in puzzles_to_test], we give each test a readable name.
@pytest.mark.parametrize(
    "puzzle_data, expected_solution, puzzle_id",
    puzzles_to_test,
    ids=[p[2] for p in puzzles_to_test],
)
def test_cp_sat_solver(puzzle_data, expected_solution, puzzle_id):
    """
    Tests the CP-SAT solver against the standard set of puzzles from conftest.py.
    """
    logger.info(f"--- [CP-SAT Solver] Starting test for: {puzzle_id} ---")
    logger.debug(f"Puzzle input: {puzzle_data}")

    actual = solve_puzzle_cp(puzzle_data)
    logger.info(f"Solution found for {puzzle_id} via CP-SAT: {actual}")

    # The CP-SAT solver might find a different but valid path if multiple solutions exist.
    # For now, we will assert equality, but in the future, we might need a more
    # robust validation function that checks the path's validity instead of equality.
    if expected_solution is not None:
        assert (
            actual == expected_solution
        ), f"[CP-SAT Solver] returned the wrong solution for {puzzle_id}."
    else:
        assert (
            actual is not None
        ), f"[CP-SAT Solver] did not find a solution for {puzzle_id}."

        grid = puzzle_data["grid"]
        blocked_cells = puzzle_data.get("blocked_cells", set())
        expected_len = len(grid) * len(grid[0]) - len(blocked_cells)
        assert (
            len(actual) == expected_len
        ), f"[CP-SAT Solver] Path length for {puzzle_id} should be {expected_len}, but was {len(actual)}."

    logger.info(f"--- [CP-SAT Solver] Finished test for: {puzzle_id} ---")
