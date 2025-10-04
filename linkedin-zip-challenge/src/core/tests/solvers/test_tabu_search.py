# src/core/tests/solvers/test_tabu_search.py

from src.core.solvers.tabu_search import solve_puzzle_tabu_search
from src.core.tests.conftest import puzzle_01_data


def test_solve_puzzle_tabu_search_returns_valid_path():
    """
    Tests that the Tabu Search solver returns a valid path format.
    """
    # Act
    # Use a small number of iterations to keep the test fast.
    solution = solve_puzzle_tabu_search(
        puzzle_01_data, num_iterations=10, tabu_list_size=5, neighborhood_size=10
    )

    # Assert
    assert solution is not None, "Solver returned None"
    assert isinstance(solution, list), "Solution is not a list"
    assert all(
        isinstance(item, tuple) for item in solution
    ), "Not all items in the solution are tuples"
