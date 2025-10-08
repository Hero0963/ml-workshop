# src/core/tests/test_puzzle_generator.py
from loguru import logger
from src.core.puzzle_generator import generate_puzzle


def test_generate_puzzle_smoke():
    """A simple smoke test to ensure the generator runs and produces a valid pair."""
    puzzle, solution_path = generate_puzzle(
        m=5, n=5, has_walls=True, num_blocked_cells=2
    )

    assert puzzle is not None
    assert solution_path is not None

    # 1. Verify path length
    m, n = puzzle["grid_size"]
    num_blocked = len(puzzle["blocked_cells"])
    assert num_blocked == 2
    assert len(solution_path) == (m * n) - num_blocked
    assert len(set(solution_path)) == len(
        solution_path
    ), "Path should not have duplicates"

    # 2. Verify waypoints are on the path and in order
    sorted_waypoints = sorted(puzzle["num_map"].items())
    last_idx = -1
    for num, pos in sorted_waypoints:
        assert (
            pos in solution_path
        ), f"Waypoint {num} at {pos} is not in the solution path"
        current_idx = solution_path.index(pos)
        assert current_idx > last_idx, f"Waypoint {num} is out of order"
        last_idx = current_idx

    # 3. Verify blocked cells are not in the path
    for blocked_cell in puzzle["blocked_cells"]:
        assert (
            blocked_cell not in solution_path
        ), "Path should not go through blocked cells"
        assert puzzle["puzzle_layout"][blocked_cell[0]][blocked_cell[1]] == "xx"

    # 4. Verify walls are not part of the path
    solution_edges = set()
    for i in range(len(solution_path) - 1):
        solution_edges.add(tuple(sorted((solution_path[i], solution_path[i + 1]))))

    assert len(puzzle["walls"]) > 0, "has_walls=True should generate walls"
    for wall in puzzle["walls"]:
        assert wall not in solution_edges, "Solution path should not cross a wall"

    logger.success("--- Puzzle Generation Smoke Test Passed ---")
    logger.info(
        f"Generated a {m}x{n} puzzle with {len(puzzle['blocked_cells'])} blocked cells and {len(puzzle['walls'])} walls."
    )
    logger.info(f"Solution path length: {len(solution_path)}")


def test_generate_puzzle_default_waypoints():
    """Tests that the default waypoint generation logic is dynamic."""
    # Use a fixed size for predictable path length
    m, n = 6, 6
    # Call without num_waypoints to test the default logic
    result = generate_puzzle(m=m, n=n, num_blocked_cells=0)

    assert result is not None
    puzzle, solution_path = result

    # The path should cover all cells
    assert len(solution_path) == m * n

    num_waypoints = len(puzzle["num_map"])
    min_expected = len(solution_path) // 4
    max_expected = len(solution_path) // 3

    # Ensure min/max are at least 2
    if max_expected < 2:
        max_expected = 2
    if min_expected < 2:
        min_expected = 2

    assert min_expected <= num_waypoints <= max_expected
    logger.success("--- Default Waypoint Test Passed ---")
    logger.info(
        f"Path length: {len(solution_path)}, Waypoints: {num_waypoints} (Expected between {min_expected}-{max_expected})"
    )
