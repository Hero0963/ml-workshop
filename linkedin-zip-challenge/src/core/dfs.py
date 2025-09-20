from typing import Dict, List, Optional, Set, Tuple

# Input Type Definition:
# The puzzle is a dictionary with the following keys:
# 'grid': A List[List[int]] representing the grid. 0 for empty, >0 for numbers.
# 'walls': A Set[Tuple[Tuple[int, int], Tuple[int, int]]] where each inner tuple
#          is a coordinate (row, col), representing blocked passages.
#          The coordinate pair in the tuple should be sorted to ensure uniqueness.

# Output Type Definition:
# The solution is a List[Tuple[int, int]], representing the path of coordinates
# from start to finish. Returns None if no solution is found.


def solve_puzzle(puzzle: Dict) -> Optional[List[Tuple[int, int]]]:
    """
    Solves a Zip puzzle using a backtracking DFS algorithm.

    Args:
        puzzle: A dictionary containing the 'grid' and 'walls'.

    Returns:
        A list of coordinates representing the solution path, or None.
    """
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    height = len(grid)
    width = len(grid[0])
    total_cells = height * width

    num_map = {
        grid[r][c]: (r, c)
        for r in range(height)
        for c in range(width)
        if grid[r][c] > 0
    }

    if 1 not in num_map:
        return None  # No starting point

    start_pos = num_map[1]
    path = [start_pos]
    visited = {start_pos}

    return _backtrack(path, visited, grid, walls, total_cells, num_map)


def _backtrack(
    path: List[Tuple[int, int]],
    visited: Set[Tuple[int, int]],
    grid: List[List[int]],
    walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    total_cells: int,
    num_map: Dict[int, Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """Recursive helper function for the backtracking algorithm."""

    if len(path) == total_cells:
        return path

    last_pos = path[-1]
    r, c = last_pos
    height = len(grid)
    width = len(grid[0])

    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Try all four directions
        nr, nc = r + dr, c + dc
        new_pos = (nr, nc)

        # Check if the move is valid
        if not (0 <= nr < height and 0 <= nc < width):
            continue  # Out of bounds
        if new_pos in visited:
            continue  # Already visited

        # Check for walls
        # Ensure the wall pair is sorted before checking existence
        wall_pair = tuple(sorted((last_pos, new_pos)))
        if wall_pair in walls:
            continue

        # Pre-move validation: If we are about to step on a numbered cell,
        # check if it's the correct one in the sequence.
        num_at_new_pos = grid[nr][nc]
        if num_at_new_pos > 0:
            # Find the highest number we have visited so far in the path
            max_visited_num = 0
            for num, pos in num_map.items():
                if pos in visited:
                    if num > max_visited_num:
                        max_visited_num = num

            # The number we are stepping on must be the next in sequence
            if num_at_new_pos != max_visited_num + 1:
                continue  # Invalid move, prune this branch

        path.append(new_pos)
        visited.add(new_pos)

        solution = _backtrack(path, visited, grid, walls, total_cells, num_map)
        if solution:
            return solution

        # Backtrack
        visited.remove(new_pos)
        path.pop()

    return None
