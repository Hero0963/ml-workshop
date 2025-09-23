# src/core/tests/test_utils.py
from src.core.utils import parse_puzzle_layout


def test_parse_puzzle_layout_real():
    """Tests the parser with a more realistic puzzle layout."""
    test_layout = [
        ["  ", "01", "xx"],
        ["02", "  ", "  "],
        ["  ", "03", "xx"],
    ]
    expected_grid = [[0, 1, 0], [2, 0, 0], [0, 3, 0]]
    expected_blocked = {(0, 2), (2, 2)}

    actual_data = parse_puzzle_layout(test_layout)

    assert actual_data["grid"] == expected_grid
    assert actual_data["blocked_cells"] == expected_blocked
