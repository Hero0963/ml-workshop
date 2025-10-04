# src/core/tests/solvers/test_genetic_algorithm.py
import pytest

from src.core.solvers.genetic_algorithm import solve_puzzle_genetic_algorithm
from src.core.tests.conftest import puzzle_01_data, puzzle_03_data


@pytest.mark.parametrize(
    "puzzle_data, puzzle_name",
    [
        (puzzle_01_data, "puzzle_01"),
        (puzzle_03_data, "puzzle_03"),
    ],
)
def test_solve_puzzle_genetic_algorithm_returns_valid_path(
    puzzle_data: dict, puzzle_name: str
) -> None:
    """
    Tests that the Genetic Algorithm solver returns a valid path format.
    It does not check for correctness, only that it produces a result.
    """
    # Act
    solution = solve_puzzle_genetic_algorithm(
        puzzle_data, population_size=20, num_generations=10, num_elites=4
    )

    # Assert
    assert solution is not None, f"Solver returned None for {puzzle_name}"
    assert isinstance(solution, list), f"Solution for {puzzle_name} is not a list"
    assert all(
        isinstance(item, tuple) for item in solution
    ), f"Not all items in the solution for {puzzle_name} are tuples"
