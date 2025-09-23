# src/core/utils.py
import os
import sys
import time
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

# ANSI escape codes for styling
BG_HIGHLIGHT = "\u001b[47;30m"  # Black text on a grey background
RESET_STYLE = "\u001b[0m"


def parse_puzzle_layout(grid_layout: list[list[str]]) -> dict[str, Any]:
    """
    Parses a grid defined with 2-character string elements ('  ', 'xx', '01', '12')
    into the standard puzzle dictionary format.
    """
    grid: list[list[int]] = []
    blocked_cells: set[tuple[int, int]] = set()
    for r, row_list in enumerate(grid_layout):
        row: list[int] = []
        for c, item in enumerate(row_list):
            if item.isdigit():
                row.append(int(item))
            elif item == "xx":
                row.append(0)
                blocked_cells.add((r, c))
            else:
                row.append(0)
        grid.append(row)
    return {"grid": grid, "blocked_cells": blocked_cells}


def visualize_solution_simple(
    puzzle_data: dict[str, Any], solution_path: list[tuple[int, int]] | None
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
    puzzle_data: dict[str, Any], solution_path: list[tuple[int, int]] | None
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
    puzzle_data: dict[str, Any],
    solution_path: list[tuple[int, int]] | None,
    visualization_func: Callable[
        [dict[str, Any], list[tuple[int, int]] | None], list[list[str]]
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
    puzzle_data: dict[str, Any],
    solution_path: list[tuple[int, int]] | None,
    speed: float = 0.25,
) -> None:
    _animate(puzzle_data, solution_path, visualize_solution_simple, speed)


def animate_solution_highlight(
    puzzle_data: dict[str, Any],
    solution_path: list[tuple[int, int]] | None,
    speed: float = 0.25,
) -> None:
    _animate(puzzle_data, solution_path, visualize_solution_highlight, speed)


def save_animation_as_gif(
    puzzle_data: dict[str, Any],
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
    height = len(grid)
    width = len(grid[0])
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


if __name__ == "__main__":
    from src.core.tests.conftest import puzzle_05_data, solution_05

    sample_puzzle_data = puzzle_05_data
    sample_soltution = solution_05

    def _simple_example():
        # Generate the GIF
        save_animation_as_gif(sample_puzzle_data, sample_soltution)

        # # Run console animations
        # print("--- Running Animation: Style 1 (Simple) ---")
        # animate_solution_simple(sample_puzzle_data, sample_soltution)

        # input("\nAnimation finished. Press Enter to run the next animation style...")

        # print("--- Running Animation: Style 2 (Highlight) ---")
        # animate_solution_highlight(sample_puzzle_data, sample_soltution)

        print("\nAll animations complete.")

    _simple_example()
