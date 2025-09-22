# src/core/tests/test_utils.py
from src.core.utils import parse_puzzle_layout, visualize_solution


def test_parse_puzzle_layout():
    """
    Tests that the parser correctly converts a 2-char layout into the standard
    puzzle data structure.
    """
    test_layout = [
        ["01", "  ", "xx"],
        ["03", "02", "  "],
    ]
    expected_data = {
        "grid": [
            [1, 0, 0],
            [3, 2, 0],
        ],
        "blocked_cells": {(0, 2)},
    }
    actual_data = parse_puzzle_layout(test_layout)
    assert actual_data["grid"] == expected_data["grid"]
    assert actual_data["blocked_cells"] == expected_data["blocked_cells"]


def test_visualize_solution():
    """
    Tests that the solution visualizer correctly creates a numbered grid
    from a solution path.
    """
    # Dummy puzzle data just to provide grid dimensions
    puzzle_data = {
        "grid": [
            [0, 0, 0],
            [0, 0, 0],
        ]
    }
    solution_path = [(0, 0), (0, 1), (1, 1), (1, 2)]
    expected_visualization = [
        ["00", "01", "  "],
        ["  ", "02", "03"],
    ]
    actual_visualization = visualize_solution(puzzle_data, solution_path)
    assert actual_visualization == expected_visualization


def test_visualize_solution_empty_path():
    """
    Tests that the visualizer returns an empty list for an empty solution path.
    """
    puzzle_data = {
        "grid": [
            [0, 0],
            [0, 0],
        ]
    }
    solution_path = []
    actual_visualization = visualize_solution(puzzle_data, solution_path)
    assert actual_visualization == []
