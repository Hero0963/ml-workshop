# src/core/tests/conftest.py

# This list holds all puzzle data for parametrized tests.
# It is imported by test files.
# The format for each item is a tuple: (puzzle_dictionary, test_id)
puzzles_to_test = [
    # Puzzle 01 Data (Corrected by user)
    (
        {
            "grid": [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 2, 0],
                [0, 0, 3, 4, 0, 0],
                [0, 0, 6, 5, 0, 0],
                [0, 8, 0, 0, 7, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            "walls": {
                tuple(sorted(((1, 0), (1, 1)))),
                tuple(sorted(((0, 1), (1, 1)))),
                tuple(sorted(((2, 1), (2, 2)))),
                tuple(sorted(((1, 3), (2, 3)))),
                tuple(sorted(((1, 4), (2, 4)))),
                tuple(sorted(((3, 3), (3, 4)))),
                tuple(sorted(((3, 1), (4, 1)))),
                tuple(sorted(((3, 2), (4, 2)))),
                tuple(sorted(((4, 4), (4, 5)))),
                tuple(sorted(((4, 4), (5, 4)))),
            },
        },
        "puzzle_01",
    ),
    # Puzzle 02 Data
    (
        {
            "grid": [
                [0, 0, 5, 8, 0, 0],
                [0, 0, 12, 0, 9, 0],
                [4, 0, 0, 0, 0, 1],
                [0, 0, 6, 7, 0, 0],
                [0, 11, 0, 0, 10, 0],
                [3, 0, 0, 0, 0, 2],
            ],
            "walls": set(),
        },
        "puzzle_02",
    ),
    # Puzzle 03 Data
    (
        {
            "grid": [
                [12, 0, 11, 0, 9, 0],
                [2, 0, 1, 0, 10, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 8, 0, 7, 0, 6],
                [0, 3, 0, 4, 0, 5],
            ],
            "walls": {
                tuple(sorted(((2, 1), (3, 1)))),
                tuple(sorted(((2, 2), (3, 2)))),
                tuple(sorted(((2, 3), (3, 3)))),
                tuple(sorted(((2, 4), (3, 4)))),
            },
        },
        "puzzle_03",
    ),
    # Puzzle 04 Data
    (
        {
            "grid": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 7, 0, 6, 0, 10, 0],
                [0, 0, 8, 0, 9, 0, 0],
                [0, 12, 0, 0, 0, 11, 0],
                [0, 0, 3, 0, 2, 0, 0],
                [0, 4, 0, 5, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "walls": {
                tuple(sorted(((0, 3), (1, 3)))),
                tuple(sorted(((1, 3), (2, 3)))),
                tuple(sorted(((2, 1), (3, 1)))),
                tuple(sorted(((2, 2), (3, 2)))),
                tuple(sorted(((2, 3), (3, 3)))),
                tuple(sorted(((2, 4), (3, 4)))),
                tuple(sorted(((2, 5), (3, 5)))),
                tuple(sorted(((3, 1), (4, 1)))),
                tuple(sorted(((3, 2), (4, 2)))),
                tuple(sorted(((3, 3), (4, 3)))),
                tuple(sorted(((3, 4), (4, 4)))),
                tuple(sorted(((3, 5), (4, 5)))),
                tuple(sorted(((4, 3), (5, 3)))),
                tuple(sorted(((5, 3), (6, 3)))),
            },
        },
        "puzzle_04",
    ),
    # Puzzle 05 Data
    (
        {
            "grid": [
                [12, 11, 0, 0, 0, 0],
                [0, 10, 0, 0, 0, 0],
                [9, 8, 7, 0, 0, 0],
                [0, 0, 0, 5, 4, 0],
                [0, 0, 0, 0, 3, 0],
                [0, 0, 0, 6, 1, 2],
            ],
            "walls": {
                tuple(sorted(((1, 3), (1, 4)))),
                tuple(sorted(((1, 4), (1, 5)))),
                tuple(sorted(((4, 0), (4, 1)))),
                tuple(sorted(((4, 1), (4, 2)))),
            },
        },
        "puzzle_05",
    ),
    # Puzzle 06 Data
    (
        {
            "grid": [
                [0, 0, 0, 0, 0, 0, 0],
                [4, 0, 5, 17, 0, 18, 6],
                [3, 0, 0, 16, 0, 0, 7],
                [13, 0, 2, 1, 0, 19, 8],
                [12, 0, 15, 0, 0, 0, 9],
                [11, 0, 14, 21, 0, 20, 10],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "walls": set(),
        },
        "puzzle_06",
    ),
]
