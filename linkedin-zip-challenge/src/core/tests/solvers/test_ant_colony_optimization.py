# src/core/tests/solvers/test_ant_colony_optimization.py

from src.core.solvers.ant_colony_optimization import solve_puzzle_ant_colony
from src.core.tests.conftest import puzzle_01_data


def test_solve_puzzle_ant_colony_returns_valid_path():
    """
    Tests that the ACO solver returns a valid path format.
    It does not check for correctness, only that it produces a result.
    """
    # Act
    # Use a small number of ants/iterations to keep the test fast.
    solution = solve_puzzle_ant_colony(puzzle_01_data, num_iterations=2, num_ants=5)

    # Assert
    assert solution is not None, "Solver returned None"
    assert isinstance(solution, list), "Solution is not a list"
    assert all(
        isinstance(item, tuple) for item in solution
    ), "Not all items in the solution are tuples"
