# src/core/tests/solvers/test_simulated_annealing.py
import pytest

from src.core.solvers.simulated_annealing import solve_puzzle_simulated_annealing
from src.core.tests.conftest import puzzle_01_data, puzzle_03_data


@pytest.mark.parametrize(
    "puzzle_data, puzzle_name",
    [
        (puzzle_01_data, "puzzle_01"),
        (puzzle_03_data, "puzzle_03"),
    ],
)
def test_solve_puzzle_simulated_annealing_returns_valid_path(
    puzzle_data: dict, puzzle_name: str
) -> None:
    """
    Tests that the Simulated Annealing solver returns a valid path format.
    It does not check for correctness, only that it produces a result.
    """
    # Act
    solution = solve_puzzle_simulated_annealing(
        puzzle_data, initial_temp=1000, final_temp=1, cooling_rate=0.9
    )

    # Assert
    assert solution is not None, f"Solver returned None for {puzzle_name}"
    assert isinstance(solution, list), f"Solution for {puzzle_name} is not a list"
    assert all(
        isinstance(item, tuple) for item in solution
    ), f"Not all items in the solution for {puzzle_name} are tuples"
