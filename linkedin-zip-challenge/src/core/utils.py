# src/core/utils.py


def parse_puzzle_layout(grid_layout: list[list[str]]) -> dict:
    """
    Parses a grid defined with 2-character string elements ('  ', 'xx', '01', '12')
    into the standard puzzle dictionary format.

    Args:
        grid_layout: A list of lists of strings representing the puzzle layout.
                     '  ' for empty, 'xx' for blocked, 2-digit strings for waypoints.

    Returns:
        A dictionary with "grid" (list[list[int]]) and "blocked_cells" (set).
    """
    grid = []
    blocked_cells = set()
    for r, row_list in enumerate(grid_layout):
        row = []
        for c, item in enumerate(row_list):
            if item.isdigit():
                row.append(int(item))
            elif item == "xx":
                row.append(0)
                blocked_cells.add((r, c))
            else:  # Assuming '  ' or other non-matching strings are empty
                row.append(0)
        grid.append(row)
    return {"grid": grid, "blocked_cells": blocked_cells}


def visualize_solution(
    puzzle_data: dict, solution_path: list[tuple[int, int]]
) -> list[list[str]]:
    """
    Creates a grid visualizing the solution path with step numbers.

    Args:
        puzzle_data: The original puzzle dictionary to get dimensions.
        solution_path: The list of coordinates representing the solution path.

    Returns:
        A new grid (list of lists of strings) with the path visualized.
    """
    if not solution_path:
        return []

    height = len(puzzle_data["grid"])
    width = len(puzzle_data["grid"][0])

    # Create an empty grid
    vis_grid = [["  " for _ in range(width)] for _ in range(height)]

    # Fill in the path
    for i, (r, c) in enumerate(solution_path):
        # Ensure the coordinate is within bounds before placing
        if 0 <= r < height and 0 <= c < width:
            vis_grid[r][c] = f"{i:02d}"  # Format number to be 2 digits

    return vis_grid


def print_grid(grid: list[list[str]]):
    """Prints a grid to the console in a readable format."""
    if not grid:
        print("Grid is empty.")
        return
    for row in grid:
        print(" ".join(row))
