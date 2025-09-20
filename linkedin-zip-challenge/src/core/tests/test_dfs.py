import os
import sys
from datetime import datetime, timezone

import pytest
from loguru import logger

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.core.dfs import solve_puzzle

# --- Logger Configuration (Final) ---
# This setup runs once when the test module is loaded.

# 1. Create a reports directory.
report_dir = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(report_dir, exist_ok=True)

# 2. Generate a UTC timestamp for the filename, including microseconds.
utc_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
log_filename = f"log_{utc_timestamp}.log"
log_file_path = os.path.join(report_dir, log_filename)

# 3. Configure logger.
logger.remove()  # Remove default handler.
# Add a handler for console output (shows INFO and above).
logger.add(sys.stderr, level="INFO")
# Add a handler to write to the timestamped log file with synchronous writing.
logger.add(
    log_file_path,
    level="DEBUG",
    enqueue=False,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS Z} | {level: <8} | {name}:{function}:{line} - {message}",
)
# ---


@pytest.fixture(autouse=True)
def log_test_name(request):
    """Automatically log the name of the test being run."""
    logger.info(f"--- Starting test: {request.node.name} ---")
    yield
    logger.info(f"--- Finished test: {request.node.name} ---\n")


def test_simple_puzzle():
    """Tests a simple 3x3 puzzle that has a known solution."""
    puzzle = {"grid": [[1, 0, 0], [0, 2, 0], [0, 0, 3]], "walls": set()}
    logger.debug(f"Puzzle input: {puzzle}")
    solution = solve_puzzle(puzzle)
    logger.info(f"Solution found: {solution}")

    assert solution is not None
    assert len(solution) == 9
    assert solution[0] == (0, 0)
    assert solution.index((1, 1)) > solution.index((0, 0))
    assert solution.index((2, 2)) > solution.index((1, 1))


def test_no_solution_due_to_wall():
    """Tests a puzzle where a wall makes the solution impossible."""
    puzzle = {
        "grid": [[1, 2]],
        "walls": {tuple(sorted(((0, 0), (0, 1))))},  # Wall between the only two cells
    }
    logger.debug(f"Puzzle input: {puzzle}")
    solution = solve_puzzle(puzzle)
    logger.info(f"Solution found: {solution}")

    assert solution is None


def test_complex_spiral_path():
    """FINAL: Tests a complex puzzle with a known, unique spiral solution."""
    puzzle = {"grid": [[1, 8, 7], [2, 0, 6], [3, 4, 5]], "walls": set()}
    expected_solution = [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
        (1, 1),
    ]
    logger.debug(f"Puzzle input: {puzzle}")
    solution = solve_puzzle(puzzle)
    logger.info(f"Solution found: {solution}")

    assert solution == expected_solution


def test_puzzle_with_wall_and_solution():
    """Tests a puzzle that has a wall but is still solvable."""
    puzzle = {"grid": [[1, 0], [3, 2]], "walls": {tuple(sorted(((0, 0), (1, 0))))}}
    logger.debug(f"Puzzle input: {puzzle}")
    solution = solve_puzzle(puzzle)
    logger.info(f"Solution found: {solution}")

    assert solution is not None
    assert solution == [(0, 0), (0, 1), (1, 1), (1, 0)]
