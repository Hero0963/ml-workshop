# src/core/utils.py
import os
import sys
import time
from typing import Callable, TypedDict, Set, List, Tuple, Dict

import random
from PIL import Image, ImageDraw, ImageFont

# ANSI escape codes for styling
BG_HIGHLIGHT = "\u001b[47;30m"  # Black text on a grey background
RESET_STYLE = "\u001b[0m"


class Puzzle(TypedDict):
    grid_size: Tuple[int, int]
    puzzle_layout: List[List[str]]
    walls: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
    grid: List[List[int]]
    blocked_cells: Set[Tuple[int, int]]
    num_map: Dict[int, Tuple[int, int]]
    start: Tuple[int, int] | None
    end: Tuple[int, int] | None


def parse_puzzle_layout(grid_layout: list[list[str]]) -> Puzzle:
    """
    Parses a grid defined with 2-character string elements ('  ', 'xx', '01', '12')
    into the standard puzzle dictionary format.
    Also creates the 'num_map' for quick waypoint lookup.
    """
    grid: list[list[int]] = []
    blocked_cells: set[tuple[int, int]] = set()
    num_map: dict[int, tuple[int, int]] = {}
    height, width = len(grid_layout), len(grid_layout[0]) if grid_layout else 0

    for r in range(height):
        row: list[int] = []
        for c in range(width):
            item = grid_layout[r][c]
            if item.isdigit():
                num = int(item)
                row.append(num)
                num_map[num] = (r, c)
            elif item == "xx":
                row.append(0)
                blocked_cells.add((r, c))
            else:
                row.append(0)
        grid.append(row)

    # Sort the num_map by waypoint number
    sorted_num_map = dict(sorted(num_map.items()))

    return {
        "grid": grid,
        "blocked_cells": blocked_cells,
        "num_map": sorted_num_map,
        "grid_size": (height, width),
        "puzzle_layout": grid_layout,
        "walls": set(),  # Initially empty, added later
        "start": sorted_num_map.get(1),
        "end": sorted_num_map.get(max(sorted_num_map.keys()) if sorted_num_map else 1),
    }


def _calculate_perfect_score(puzzle: Puzzle) -> int:
    """Calculates the theoretical maximum score for a perfect solution."""
    grid = puzzle["grid"]
    blocked_cells = puzzle.get("blocked_cells", set())
    num_map = puzzle["num_map"]
    height = len(grid)
    width = len(grid[0])
    visitable_cells = (height * width) - len(blocked_cells)

    # 1. Path length reward
    score = visitable_cells * 10

    # 2. Waypoint rewards
    if num_map:
        num_waypoints = len(num_map)
        # Sum of an arithmetic series: n * (a1 + an) / 2
        # Here: num_waypoints * (1 + num_waypoints) / 2
        waypoint_bonus = sum(range(1, num_waypoints + 1))
        score += 20000 * waypoint_bonus

    # 3. Jackpot
    score += 1_000_000

    return score


def calculate_fitness_score(
    puzzle: Puzzle, path: list[tuple[int, int]]
) -> tuple[int, int]:
    """
    Calculates a fitness score for a given path against a puzzle.
    Higher scores are better. The score is designed to guide metaheuristic search.

    Returns:
        A tuple of (current_score, perfect_score).
    """
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    num_map = puzzle["num_map"]
    height = len(grid)
    width = len(grid[0])
    visitable_cells = (height * width) - len(blocked_cells)

    perfect_score = _calculate_perfect_score(puzzle)
    score = 0

    # --- Hard Constraint Penalties (for invalid paths) ---
    if len(path) != len(set(path)) or not path or path[0] in blocked_cells:
        return -1, perfect_score

    for i in range(len(path) - 1):
        # Adjacency check: Ensure path doesn't "jump" between non-neighboring cells.
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        manhattan_distance = abs(r1 - r2) + abs(c1 - c2)
        if manhattan_distance > 1:
            score -= 100000  # Heavy penalty for jumping

        # Wall check
        wall_pair = tuple(sorted((path[i], path[i + 1])))
        if wall_pair in walls:
            score -= 50000

    # --- Soft Constraint Rewards/Penalties (for path quality) ---
    score += len(path) * 10

    next_waypoint_num = 1
    if 1 in num_map and path[0] != num_map[1]:
        score -= 100000
    elif 1 in num_map and path[0] == num_map[1]:
        score += 20000  # Correct start reward is not scaled
        next_waypoint_num = 2

    for pos in path[1:]:
        cell_value = grid[pos[0]][pos[1]]
        if cell_value > 0:
            if cell_value == next_waypoint_num:
                score += 20000 * next_waypoint_num
                next_waypoint_num += 1
            else:
                score -= 5000

    # --- Jackpot for a complete, valid solution ---
    all_waypoints_visited = not num_map or next_waypoint_num > max(num_map.keys())
    if len(path) == visitable_cells and all_waypoints_visited and score > 0:
        score += 1_000_000

    return score, perfect_score


def visualize_solution_simple(
    puzzle_data: Puzzle, solution_path: list[tuple[int, int]] | None
) -> list[list[str]]:
    """Visualizes the path using brackets `[]` and asterisks `*`."""
    grid: list[list[int]] = puzzle_data["grid"]
    blocked_cells: set[tuple[int, int]] = puzzle_data.get("blocked_cells", set())
    height = len(grid)
    width = len(grid[0])
    vis_grid = [["." for _ in range(width)] for _ in range(height)]
    path_coords = set(solution_path) if solution_path else set()

    for r in range(height):
        for c in range(width):
            pos = (r, c)
            waypoint = grid[r][c]
            content = str(waypoint) if waypoint > 0 else "."

            if pos in blocked_cells:
                vis_grid[r][c] = "XXX"
                continue

            if pos in path_coords:
                if waypoint > 0:
                    vis_grid[r][c] = f"[{content}]"
                else:
                    vis_grid[r][c] = " * "
            else:
                vis_grid[r][c] = content
    return vis_grid


def visualize_solution_highlight(
    puzzle_data: Puzzle, solution_path: list[tuple[int, int]] | None
) -> list[list[str]]:
    """Visualizes the path using ANSI background color highlighting."""
    grid: list[list[int]] = puzzle_data["grid"]
    blocked_cells: set[tuple[int, int]] = puzzle_data.get("blocked_cells", set())
    height = len(grid)
    width = len(grid[0])
    vis_grid = [["" for _ in range(width)] for _ in range(height)]
    path_coords = set(solution_path) if solution_path else set()

    for r in range(height):
        for c in range(width):
            pos = (r, c)
            content = str(grid[r][c]) if grid[r][c] > 0 else "."
            if pos in blocked_cells:
                content = "XXX"

            if pos in path_coords:
                vis_grid[r][c] = f"{BG_HIGHLIGHT}{content}{RESET_STYLE}"
            else:
                vis_grid[r][c] = content
    return vis_grid


def print_grid(grid: list[list[str]], col_widths: list[int] | None = None) -> None:
    """Prints a grid to the console with aligned columns."""
    if not grid:
        print("Grid is empty.")
        return

    def get_visible_length(s: str) -> int:
        import re

        return len(re.sub(r"\u001b\[[0-9;]*m", "", s))

    if col_widths is None:
        num_cols = max(len(row) for row in grid) if grid else 0
        if num_cols == 0:
            return
        col_widths = [0] * num_cols
        for row in grid:
            for i, item in enumerate(row):
                vis_len = get_visible_length(item)
                if vis_len > col_widths[i]:
                    col_widths[i] = vis_len

    for row in grid:
        line: list[str] = []
        for i, item in enumerate(row):
            vis_len = get_visible_length(item)
            padding = " " * (col_widths[i] - vis_len)
            line.append(padding + item)
        print("  ".join(line))


def _animate(
    puzzle_data: Puzzle,
    solution_path: list[tuple[int, int]] | None,
    visualization_func: Callable[
        [Puzzle, list[tuple[int, int]] | None], list[list[str]]
    ],
    speed: float,
) -> None:
    """Generic animation loop with fixed grid layout."""
    if not solution_path:
        print("No solution path to animate.")
        return

    final_grid = visualization_func(puzzle_data, solution_path)

    def get_visible_length(s: str) -> int:
        import re

        return len(re.sub(r"\u001b\[[0-9;]*m", "", s))

    num_cols = max(len(row) for row in final_grid) if final_grid else 0
    col_widths: list[int] = [0] * num_cols
    for row in final_grid:
        for i, item in enumerate(row):
            vis_len = get_visible_length(item)
            if vis_len > col_widths[i]:
                col_widths[i] = vis_len

    for i in range(len(solution_path) + 1):
        partial_path = solution_path[:i]
        vis_grid = visualization_func(puzzle_data, partial_path)
        os.system("cls" if sys.platform == "win32" else "clear")
        print(f"--- Animation Style: {visualization_func.__name__} ---")
        print(f"Step {i}/{len(solution_path)}")
        print_grid(vis_grid, col_widths=col_widths)  # Pass pre-calculated widths
        time.sleep(speed)


def animate_solution_simple(
    puzzle_data: Puzzle,
    solution_path: list[tuple[int, int]] | None,
    speed: float = 0.25,
) -> None:
    _animate(puzzle_data, solution_path, visualize_solution_simple, speed)


def animate_solution_highlight(
    puzzle_data: Puzzle,
    solution_path: list[tuple[int, int]] | None,
    speed: float = 0.25,
) -> None:
    _animate(puzzle_data, solution_path, visualize_solution_highlight, speed)


def save_animation_as_gif(
    puzzle_data: Puzzle,
    solution_path: list[tuple[int, int]] | None,
    filename: str = "solution.gif",
    speed: int = 250,
) -> None:
    """Generates and saves the solution animation as a GIF file."""
    if not solution_path:
        print("No solution path to generate GIF.")
        return

    grid: list[list[int]] = puzzle_data["grid"]
    walls: set[tuple[tuple[int, int], tuple[int, int]]] = puzzle_data.get(
        "walls", set()
    )
    blocked_cells: set[tuple[int, int]] = puzzle_data.get("blocked_cells", set())
    height, width = puzzle_data["grid_size"]
    cell_size = 50
    margin = 10
    img_width = width * cell_size + 2 * margin
    img_height = height * cell_size + 2 * margin

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    frames: list[Image.Image] = []
    for i in range(len(solution_path) + 1):
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)
        path_coords = set(solution_path[:i])

        for r in range(height):
            for c in range(width):
                x0 = margin + c * cell_size
                y0 = margin + r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                rect = [(x0, y0), (x1, y1)]

                # Draw blocked cells first
                if (r, c) in blocked_cells:
                    draw.rectangle(rect, fill="black")
                    continue

                if (r, c) in path_coords:
                    draw.rectangle(rect, fill="#cccccc")  # Highlight path

                draw.rectangle(rect, outline="black")

                content = str(grid[r][c]) if grid[r][c] > 0 else "."
                text_bbox = draw.textbbox((0, 0), content, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_pos = (
                    x0 + (cell_size - text_width) / 2,
                    y0 + (cell_size - text_height) / 2,
                )
                draw.text(text_pos, content, fill="black", font=font)

        # Draw walls
        for wall in walls:
            (r1, c1), (r2, c2) = wall
            # Horizontal wall
            if r1 == r2:
                line_x0 = margin + max(c1, c2) * cell_size
                line_y0 = margin + r1 * cell_size
                line_x1 = line_x0
                line_y1 = line_y0 + cell_size
                draw.line(
                    [(line_x0, line_y0), (line_x1, line_y1)], fill="black", width=3
                )
            # Vertical wall
            else:
                line_x0 = margin + c1 * cell_size
                line_y0 = margin + max(r1, r2) * cell_size
                line_x1 = line_x0 + cell_size
                line_y1 = line_y0
                draw.line(
                    [(line_x0, line_y0), (line_x1, line_y1)], fill="black", width=3
                )

        frames.append(img)

    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=speed,
        loop=0,
    )
    print(f"Animation saved to {filename}")


def generate_random_path(puzzle: Puzzle) -> list[tuple[int, int]]:
    """Generates a single random path through the puzzle."""
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    num_map = puzzle["num_map"]
    height, width = puzzle["grid_size"]

    if 1 not in num_map:
        return []

    start_pos = num_map[1]
    if start_pos in blocked_cells:
        return []

    path = [start_pos]
    visited = {start_pos}
    current_pos = start_pos

    while True:
        r, c = current_pos
        valid_neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            new_pos = (nr, nc)

            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if new_pos in visited:
                continue
            if new_pos in blocked_cells:
                continue
            wall_pair = tuple(sorted((current_pos, new_pos)))
            if wall_pair in walls:
                continue

            valid_neighbors.append(new_pos)

        if not valid_neighbors:
            break

        next_pos = random.choice(valid_neighbors)
        path.append(next_pos)
        visited.add(next_pos)
        current_pos = next_pos

    return path


def generate_neighbor_path(
    path: list[tuple[int, int]], puzzle: Puzzle
) -> list[tuple[int, int]]:
    """
    Generates a neighbor path using a "truncate and regrow" strategy.
    This ensures the generated path is always contiguous on the grid.
    """
    if len(path) <= 2:
        return path.copy()

    # 1. Truncate the path at a random point (but not at the very start).
    cut_index = random.randint(1, len(path) - 1)
    new_path = path[:cut_index]

    # 2. Set up the state for regrowth.
    visited = set(new_path)
    current_pos = new_path[-1]

    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    height, width = puzzle["grid_size"]

    # 3. Regrow the path with a random walk from the truncation point.
    while True:
        r, c = current_pos
        valid_neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            new_pos = (nr, nc)

            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if new_pos in visited:
                continue
            if new_pos in blocked_cells:
                continue
            wall_pair = tuple(sorted((current_pos, new_pos)))
            if wall_pair in walls:
                continue

            valid_neighbors.append(new_pos)

        if not valid_neighbors:
            break  # Stuck, no valid moves

        next_pos = random.choice(valid_neighbors)
        new_path.append(next_pos)
        visited.add(next_pos)
        current_pos = next_pos

    return new_path


if __name__ == "__main__":
    from src.core.tests.conftest import puzzle_05_data, solution_05

    def _simple_example():
        # Generate the GIF
        save_animation_as_gif(puzzle_05_data, solution_05)

        # # Run console animations
        # print("--- Running Animation: Style 1 (Simple) ---")
        # animate_solution_simple(sample_puzzle_data, sample_soltution)

        # input("\nAnimation finished. Press Enter to run the next animation style...")

        # print("--- Running Animation: Style 2 (Highlight) ---")
        # animate_solution_highlight(sample_puzzle_data, sample_soltution)

        print("\nAll animations complete.")

    _simple_example()
