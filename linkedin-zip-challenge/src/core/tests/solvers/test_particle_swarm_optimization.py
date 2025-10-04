# src/core/tests/solvers/test_particle_swarm_optimization.py

from src.core.solvers.particle_swarm_optimization import solve_puzzle_pso
from src.core.tests.conftest import puzzle_01_data


def test_solve_puzzle_pso_returns_valid_path():
    """
    Tests that the PSO solver returns a valid path format.
    """
    # Act
    # Use a small swarm and few iterations to keep the test fast.
    solution = solve_puzzle_pso(puzzle_01_data, swarm_size=10, num_iterations=5)

    # Assert
    assert solution is not None, "Solver returned None"
    assert isinstance(solution, list), "Solution is not a list"
    assert all(
        isinstance(item, tuple) for item in solution
    ), "Not all items in the solution are tuples"
