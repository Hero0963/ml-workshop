# src/core/tests/vl_models/test_dataset_builder.py
"""Pins the properties the synthetic dataset has to have to be worth training on.

The 2025-10 dataset failed on exactly these axes -- one grid size, 2-5 walls, walls the
same colour as the grid lines -- so each is asserted rather than assumed.
"""

import json
import random

import pytest

from src.core.utils import parse_puzzle_layout
from src.core.vl_models.dataset_builder import (
    build_dataset,
    draw_recipe,
    sample_walls,
    verify_dataset,
)
from src.core.vl_models.render_puzzle import (
    DARK_THEME,
    LIGHT_THEME,
    render_puzzle,
)
from src.core.vl_models.schema import SimplePuzzleOutput

SEED = 4242


def _straight_path_puzzle() -> tuple[dict, list[tuple[int, int]]]:
    """A 2x3 board whose solution snakes through every cell."""
    puzzle = parse_puzzle_layout([["01", "  ", "  "], ["  ", "  ", "06"]])
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    return puzzle, path


class TestSampleWalls:
    """Solvability is guaranteed by construction -- as long as this holds."""

    def test_never_places_a_wall_on_the_solution(self):
        puzzle, path = _straight_path_puzzle()
        solution_edges = {
            tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)
        }
        for seed in range(30):
            walls = sample_walls(puzzle, path, 2, random.Random(seed))
            assert not (walls & solution_edges)

    def test_places_the_requested_number(self):
        puzzle, path = _straight_path_puzzle()
        assert len(sample_walls(puzzle, path, 2, random.Random(1))) == 2

    def test_zero_walls_is_allowed(self):
        """Two of the six real screenshots have no walls at all."""
        puzzle, path = _straight_path_puzzle()
        assert sample_walls(puzzle, path, 0, random.Random(1)) == set()

    def test_clamps_to_the_available_edges(self):
        puzzle, path = _straight_path_puzzle()
        walls = sample_walls(puzzle, path, 999, random.Random(1))
        assert 0 < len(walls) < 999


class TestDrawRecipe:
    def test_is_deterministic_for_an_index(self):
        first = draw_recipe(7, SEED, (6,), 0, 12)
        second = draw_recipe(7, SEED, (6,), 0, 12)
        assert first == second

    def test_different_indices_differ(self):
        recipes = [draw_recipe(i, SEED, (6,), 0, 12) for i in range(40)]
        assert len({r.wall_count for r in recipes}) > 1
        assert len({r.theme for r in recipes}) == 2

    def test_respects_the_wall_range(self):
        for index in range(200):
            recipe = draw_recipe(index, SEED, (6,), 3, 5)
            assert 3 <= recipe.wall_count <= 5

    def test_respects_the_requested_sizes(self):
        for index in range(50):
            assert draw_recipe(index, SEED, (6,), 0, 12).size == 6


class TestRenderer:
    def test_walls_and_grid_lines_are_different_colours(self):
        """The defect that made the old dataset teach a cue the real UI does not have."""
        for theme in (LIGHT_THEME, DARK_THEME):
            assert theme.wall != theme.grid_line

    def test_rendering_is_deterministic(self):
        puzzle = parse_puzzle_layout([["01", "  "], ["  ", "02"]])
        puzzle["walls"] = {((0, 0), (0, 1))}
        first = render_puzzle(puzzle, cell_size=60, show_buttons=False)
        second = render_puzzle(puzzle, cell_size=60, show_buttons=False)
        assert first.tobytes() == second.tobytes()

    def test_buttons_add_height_but_not_width(self):
        puzzle = parse_puzzle_layout([["01", "  "], ["  ", "02"]])
        without = render_puzzle(puzzle, cell_size=60, show_buttons=False)
        with_buttons = render_puzzle(puzzle, cell_size=60, show_buttons=True)
        assert with_buttons.width == without.width
        assert with_buttons.height > without.height


@pytest.mark.slow
class TestBuildDataset:
    """A small end-to-end build. Kept tiny so the suite stays fast."""

    def test_writes_a_verifiable_dataset(self, tmp_path):
        dataset_dir = build_dataset(
            count=4, name="unit", seed=SEED, output_root=tmp_path, verify=False
        )
        records = [
            json.loads(line)
            for line in (dataset_dir / "metadata.jsonl").read_text("utf-8").splitlines()
        ]

        assert len(records) == 4
        for record in records:
            assert (dataset_dir / record["file_name"]).is_file()
            # The label parses as the very schema the parser validates against.
            assert SimplePuzzleOutput(**json.loads(record["label"]))
        assert verify_dataset(dataset_dir)

    def test_verification_catches_a_tampered_image(self, tmp_path):
        dataset_dir = build_dataset(
            count=2, name="tamper", seed=SEED, output_root=tmp_path, verify=False
        )
        first = json.loads(
            (dataset_dir / "metadata.jsonl").read_text("utf-8").splitlines()[0]
        )
        target = dataset_dir / first["file_name"]
        target.write_bytes(target.read_bytes() + b"tampered")

        assert not verify_dataset(dataset_dir)
