# src/core/tests/vl_models/test_schema.py
"""The label format is the inference format -- these tests are what keep it that way.

The 2025-10 dataset wrote a Chain-of-Draft label while the parser expected
``layout`` + ``walls``, so training targets never looked like what the model is asked to
emit. ``schema.py`` removes the gap; if it reopens, one of these fails.
"""

import json

from src.core.utils import parse_puzzle_layout
from src.core.vl_models.prompt_baseline import PUZZLE_01_JSON_STR
from src.core.vl_models.schema import (
    SimplePuzzleOutput,
    from_puzzle,
    to_prompt_json,
)


def _puzzle_from_example() -> dict:
    example = SimplePuzzleOutput(**json.loads(PUZZLE_01_JSON_STR))
    puzzle = parse_puzzle_layout(example.layout)
    puzzle["walls"] = {
        tuple(sorted((tuple(wall.cell1), tuple(wall.cell2)))) for wall in example.walls
    }
    return puzzle


class TestFromPuzzle:
    def test_round_trips_a_real_example(self):
        """Puzzle -> schema -> JSON must describe the same board as the example."""
        regenerated = json.loads(to_prompt_json(from_puzzle(_puzzle_from_example())))
        original = json.loads(PUZZLE_01_JSON_STR)

        assert regenerated["layout"] == original["layout"]
        assert _wall_set(regenerated) == _wall_set(original)

    def test_walls_are_canonically_ordered(self):
        """Sorted output means two runs cannot produce different label text."""
        walls = json.loads(to_prompt_json(from_puzzle(_puzzle_from_example())))["walls"]
        keys = [(w["cell1"], w["cell2"]) for w in walls]
        assert keys == sorted(keys)

    def test_blocked_cells_render_as_xx(self):
        puzzle = parse_puzzle_layout([["01", "xx"], ["  ", "02"]])
        assert from_puzzle(puzzle).layout == [["01", "xx"], ["  ", "02"]]

    def test_numbers_keep_two_digit_padding(self):
        puzzle = parse_puzzle_layout([["01", "12"]])
        assert from_puzzle(puzzle).layout == [["01", "12"]]


class TestPromptJsonFormat:
    """Format matters: it is both the demonstrated shape and the token bill."""

    def test_each_layout_row_is_one_line(self):
        text = to_prompt_json(from_puzzle(_puzzle_from_example()))
        row_lines = [
            line for line in text.splitlines() if line.strip().startswith('["')
        ]
        assert len(row_lines) == 6

    def test_line_count_matches_the_few_shot_example(self):
        text = to_prompt_json(from_puzzle(_puzzle_from_example()))
        assert len(text.splitlines()) == len(PUZZLE_01_JSON_STR.splitlines())

    def test_a_wall_free_board_uses_an_empty_array(self):
        puzzle = parse_puzzle_layout([["01", "  "], ["  ", "02"]])
        assert '"walls": []' in to_prompt_json(from_puzzle(puzzle))

    def test_output_is_valid_json(self):
        text = to_prompt_json(from_puzzle(_puzzle_from_example()))
        assert SimplePuzzleOutput(**json.loads(text))


def _wall_set(payload: dict) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    return {
        tuple(sorted((tuple(wall["cell1"]), tuple(wall["cell2"]))))
        for wall in payload["walls"]
    }
