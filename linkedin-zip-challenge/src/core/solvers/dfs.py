# src/core/solvers/dfs.py

from loguru import logger

# Input Type Definition:
# The puzzle is a dictionary with the following keys:
# 'grid': A list[list[int]] representing the grid. 0 for empty, >0 for numbers.
# 'walls': A set[tuple[tuple[int, int], tuple[int, int]]] where each inner tuple
#          is a coordinate (row, col), representing blocked passages.
#          The coordinate pair in the tuple should be sorted to ensure uniqueness.
# 'blocked_cells': A set of (row, col) tuples that cannot be entered.

# Output Type Definition:
# The solution is a list[tuple[int, int]], representing the path of coordinates
# from start to finish. Returns None if no solution is found.


def solve_puzzle(puzzle: dict) -> list[tuple[int, int]] | None:
    """
    Solves a Zip puzzle using a backtracking DFS algorithm.

    Args:
        puzzle: A dictionary containing the 'grid', 'walls', and optional 'blocked_cells'.

    Returns:
        A list of coordinates representing the solution path, or None.
    """
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    height = len(grid)
    width = len(grid[0])
    visitable_cells = (height * width) - len(blocked_cells)

    num_map = {
        grid[r][c]: (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] > 0
    }

    if not num_map:
        # Handle empty puzzles or puzzles with no numbers
        start_pos = (0, 0)
        if start_pos in blocked_cells:
            return None
        path = [start_pos]
        visited = {start_pos}
        return _backtrack(
            path, visited, grid, walls, blocked_cells, visitable_cells, num_map, 1
        )

    if 1 not in num_map:
        return None  # No starting point

    start_pos = num_map[1]
    if start_pos in blocked_cells:
        return None

    path = [start_pos]
    visited = {start_pos}

    return _backtrack(
        path, visited, grid, walls, blocked_cells, visitable_cells, num_map, 2
    )


def _backtrack(
    path: list[tuple[int, int]],
    visited: set[tuple[int, int]],
    grid: list[list[int]],
    walls: set[tuple[tuple[int, int], tuple[int, int]]],
    blocked_cells: set[tuple[int, int]],
    visitable_cells: int,
    num_map: dict[int, tuple[int, int]],
    next_waypoint: int,
) -> list[tuple[int, int]] | None:
    """Recursive helper function for the backtracking algorithm."""
    logger.debug(
        f"Enter backtrack: path_len={len(path)}, next_waypoint={next_waypoint}, last_pos={path[-1]}"
    )

    # Base case: if all visitable cells have been visited
    if len(path) == visitable_cells:
        # And all waypoints have been collected in order
        if not num_map or next_waypoint > max(num_map.keys()):
            logger.debug("SUCCESS: Path is full and all waypoints visited.")
            return path
        else:
            logger.debug(
                f"FAIL: Path is full but not all waypoints visited. Next expected: {next_waypoint}"
            )
            return None  # Path is full but didn't hit all waypoints

    last_pos = path[-1]
    r, c = last_pos
    height = len(grid)
    width = len(grid[0])

    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Try all four directions
        nr, nc = r + dr, c + dc
        new_pos = (nr, nc)

        # Standard validity checks
        if not (0 <= nr < height and 0 <= nc < width):
            continue
        if new_pos in visited:
            continue
        if new_pos in blocked_cells:
            logger.trace(f"Pruning {new_pos}: is a blocked cell.")
            continue
        wall_pair = tuple(sorted((last_pos, new_pos)))
        if wall_pair in walls:
            logger.trace(f"Pruning {new_pos}: wall detected at {wall_pair}.")
            continue

        num_at_new_pos = grid[nr][nc]

        # Waypoint logic: if we hit a numbered cell, it must be the one we're looking for
        if num_at_new_pos > 0 and num_at_new_pos != next_waypoint:
            logger.trace(
                f"Pruning {new_pos}: hit waypoint {num_at_new_pos}, expected {next_waypoint}."
            )
            continue

        path.append(new_pos)
        visited.add(new_pos)

        new_next_waypoint = (
            next_waypoint + 1 if num_at_new_pos == next_waypoint else next_waypoint
        )

        solution = _backtrack(
            path,
            visited,
            grid,
            walls,
            blocked_cells,
            visitable_cells,
            num_map,
            new_next_waypoint,
        )
        if solution:
            return solution

        # Backtrack
        visited.remove(new_pos)
        path.pop()
        logger.trace(f"Backtracking from {new_pos}. Path len: {len(path)}")

    return None
