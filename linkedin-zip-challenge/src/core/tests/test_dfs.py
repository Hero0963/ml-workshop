# src/core/tests/test_dfs.py
import os
import sys
from datetime import datetime, timezone

import pytest
from loguru import logger

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.core.dfs import solve_puzzle
from src.core.tests.conftest import puzzles_to_test

# --- Logger Configuration ---
report_dir = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(report_dir, exist_ok=True)
utc_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
log_filename = f"log_{utc_timestamp}.log"
log_file_path = os.path.join(report_dir, log_filename)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(
    log_file_path,
    level="DEBUG",
    enqueue=False,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS Z} | {level: <8} | {name}:{function}:{line} - {message}",
)
# ---


# By setting ids=[p[1] for p in puzzles_to_test], we give each test a readable name.
@pytest.mark.parametrize(
    "puzzle_data, puzzle_id", puzzles_to_test, ids=[p[1] for p in puzzles_to_test]
)
def test_image_puzzles(puzzle_data, puzzle_id):
    """
    Tests puzzles transcribed from images, with data provided from conftest.py.
    The primary goal is to run the solver and log the output for each puzzle.
    """
    logger.info(f"--- Starting test for: {puzzle_id} ---")
    logger.debug(f"Puzzle input: {puzzle_data}")

    actual = solve_puzzle(puzzle_data)
    logger.info(f"Solution found for {puzzle_id}: {actual}")

    # For now, we just assert that a solution is found and the path is complete.
    # We can add assertions for specific paths later if needed.
    assert actual is not None, f"Solver did not find a solution for {puzzle_id}."

    grid = puzzle_data["grid"]
    blocked_cells = puzzle_data.get("blocked_cells", set())
    expected_len = len(grid) * len(grid[0]) - len(blocked_cells)
    assert (
        len(actual) == expected_len
    ), f"Path length for {puzzle_id} should be {expected_len}, but was {len(actual)}."
    logger.info(f"--- Finished test for: {puzzle_id} ---\n")
