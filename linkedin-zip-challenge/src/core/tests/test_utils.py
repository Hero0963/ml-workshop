# src/core/tests/test_utils.py

import pytest

from src.core.tests.conftest import puzzle_01_data, solution_01
from src.core.utils import (
    calculate_fitness_score,
    generate_neighbor_path,
    generate_random_path,
    parse_puzzle_layout,
)


def test_parse_puzzle_layout_real():
    """Tests the parser with a more realistic puzzle layout."""
    test_layout = [
        ["  ", "01", "xx"],
        ["02", "  ", "  "],
        ["  ", "03", "xx"],
    ]
    expected_grid = [[0, 1, 0], [2, 0, 0], [0, 3, 0]]
    expected_blocked = {(0, 2), (2, 2)}
    expected_num_map = {1: (0, 1), 2: (1, 0), 3: (2, 1)}

    actual_data = parse_puzzle_layout(test_layout)

    assert actual_data["grid"] == expected_grid
    assert actual_data["blocked_cells"] == expected_blocked
    assert actual_data["num_map"] == expected_num_map


def test_calculate_fitness_score():
    """
    Tests the fitness function with various path scenarios to ensure correct scoring.
    """
    puzzle = puzzle_01_data
    perfect_path = solution_01

    # Case 1: Perfect solution -> should receive a very high score (with jackpot)
    score_perfect, perfect_score_val = calculate_fitness_score(puzzle, perfect_path)

    # Case 2: Incomplete but valid path -> should have a positive score, lower than perfect
    incomplete_path = perfect_path[:15]
    score_incomplete, _ = calculate_fitness_score(puzzle, incomplete_path)

    # Case 3: Path with a wall crossing -> should have a large penalty
    wall_cross_path = [(1, 0), (1, 1)] + perfect_path[2:]
    score_wall_cross, _ = calculate_fitness_score(puzzle, wall_cross_path)

    # Case 4: Path with a loop -> should be invalid and return -1
    looped_path = [(1, 1), (2, 1), (1, 1)] + perfect_path[3:]
    score_looped, _ = calculate_fitness_score(puzzle, looped_path)

    # Case 5: Path that doesn't start at waypoint 1 -> massive penalty
    wrong_start_path = perfect_path[1:]
    score_wrong_start, _ = calculate_fitness_score(puzzle, wrong_start_path)

    # --- Assertions ---
    # A perfect score should match the calculated perfect score and be > 1M
    assert score_perfect == perfect_score_val
    assert score_perfect > 1_000_000

    # An incomplete path should be scored positively but less than the perfect one
    assert score_incomplete > 0
    assert score_perfect > score_incomplete

    # A path crossing a wall should be heavily penalized
    assert score_wall_cross < score_incomplete

    # A path with duplicate nodes is invalid
    assert score_looped == -1

    # A path that doesn't start correctly is heavily penalized
    assert score_wrong_start < 0


def test_generate_random_path_is_valid():
    """
    Tests that generate_random_path produces a valid, contiguous path.
    """
    puzzle = puzzle_01_data
    path = generate_random_path(puzzle)

    # 1. Basic format validation
    assert path is not None
    assert len(path) > 0, "Path should not be empty"

    # 2. Must start at waypoint 1
    assert path[0] == puzzle["num_map"][1]

    # 3. All nodes must be unique
    assert len(path) == len(set(path)), "Path contains duplicate nodes"

    # 4. Path must be contiguous (no jumps)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        manhattan_distance = abs(r1 - r2) + abs(c1 - c2)
        assert (
            manhattan_distance == 1
        ), f"Path jumps between non-adjacent nodes: {path[i]} to {path[i+1]}"


def test_generate_neighbor_path_is_valid():
    """
    Tests that generate_neighbor_path produces a valid, contiguous path.
    """
    puzzle = puzzle_01_data
    # Generate a reasonably long path to mutate
    initial_path = generate_random_path(puzzle)
    if len(initial_path) < 5:
        # If the initial path is too short, try again or skip
        initial_path = generate_random_path(puzzle)
        if len(initial_path) < 5:
            pytest.skip(
                "Could not generate a long enough initial path to test mutation"
            )

    new_path = generate_neighbor_path(initial_path, puzzle)

    # 1. Basic format validation
    assert new_path is not None
    assert len(new_path) > 0, "New path should not be empty"

    # 2. Must still start at waypoint 1
    assert new_path[0] == puzzle["num_map"][1]

    # 3. All nodes must be unique
    assert len(new_path) == len(set(new_path)), "New path contains duplicate nodes"

    # 4. Path must be contiguous (no jumps)
    for i in range(len(new_path) - 1):
        r1, c1 = new_path[i]
        r2, c2 = new_path[i + 1]
        manhattan_distance = abs(r1 - r2) + abs(c1 - c2)
        assert (
            manhattan_distance == 1
        ), f"New path jumps between non-adjacent nodes: {new_path[i]} to {new_path[i+1]}"
