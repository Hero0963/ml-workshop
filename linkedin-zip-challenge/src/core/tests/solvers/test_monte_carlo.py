# src/core/tests/solvers/test_monte_carlo.py

from loguru import logger

from src.core.solvers.monte_carlo import solve_puzzle_monte_carlo
from src.core.tests.conftest import puzzle_01_data


def test_monte_carlo_solver_run():
    """
    Tests that the Monte Carlo solver runs without errors and returns a valid,
    if not optimal, path. This test verifies the basic integrity of the solver.
    """
    puzzle = puzzle_01_data
    attempts = 100  # Keep attempts low for a fast test

    logger.info("--- [Monte Carlo] Starting integrity test ---")

    # Run the solver
    solution = solve_puzzle_monte_carlo(puzzle, attempts=attempts)

    # --- Assertions ---

    # 1. The solver should return a path
    assert solution is not None, "Monte Carlo solver should find a path."

    # 2. The path should not be empty
    assert len(solution) > 0, "Returned path should not be empty."

    # 3. The path must start at waypoint 1
    assert solution[0] == puzzle["num_map"][1], "Path must start at waypoint 1."

    # 4. The path must be simple (no loops)
    assert len(solution) == len(
        set(solution)
    ), "Path should not contain duplicate cells."

    # 5. The path should not cross any walls
    walls = puzzle.get("walls", set())
    for i in range(len(solution) - 1):
        wall_pair = tuple(sorted((solution[i], solution[i + 1])))
        assert wall_pair not in walls, f"Path crosses a wall: {wall_pair}"

    logger.info("--- [Monte Carlo] Integrity test passed ---")
