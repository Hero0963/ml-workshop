# src/core/puzzle_generation/generate_cod_dataset.py

import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Finds the project root by searching upwards for a marker file."""
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(
        f"Could not find project root. Marker '{marker}' not found."
    )


# --- Add project root to sys.path for standalone execution ---
PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from PIL import Image, ImageDraw, ImageFont
    from src.core.puzzle_generation.puzzle_generator import generate_puzzle
    from src.core.utils import Puzzle
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print("Please ensure you are in the correct environment and all paths are correct.")
    exit(1)


def save_puzzle_as_png(puzzle_data: Puzzle, filename: str) -> None:
    """Generates and saves a static image of the unsolved puzzle."""
    grid: list[list[int]] = puzzle_data["grid"]
    walls = puzzle_data.get("walls", set())
    blocked_cells = puzzle_data.get("blocked_cells", set())
    height, width = puzzle_data["grid_size"]
    cell_size = 50
    margin = 10
    img_width = width * cell_size + 2 * margin
    img_height = height * cell_size + 2 * margin

    try:
        main_font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        main_font = ImageFont.load_default()

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    for r in range(height):
        for c in range(width):
            x0 = margin + c * cell_size
            y0 = margin + r * cell_size
            rect = [(x0, y0), (x0 + cell_size, y0 + cell_size)]
            pos = (r, c)

            draw.rectangle(rect, fill="white", outline="black")

            if pos in blocked_cells:
                draw.rectangle(rect, fill="black")
                continue

            content = str(grid[r][c]) if grid[r][c] > 0 else ""
            if content:
                text_bbox = draw.textbbox((0, 0), content, font=main_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_pos = (
                    x0 + (cell_size - text_width) / 2,
                    y0 + (cell_size - text_height) / 2,
                )
                draw.text(text_pos, content, fill="black", font=main_font)

    for wall in walls:
        (r1, c1), (r2, c2) = wall
        if r1 == r2:  # Vertical wall
            line_x0 = margin + max(c1, c2) * cell_size
            line_y0 = margin + r1 * cell_size
            draw.line(
                [(line_x0, line_y0), (line_x0, line_y0 + cell_size)],
                fill="black",
                width=5,
            )
        else:  # Horizontal wall
            line_x0 = margin + c1 * cell_size
            line_y0 = margin + max(r1, r2) * cell_size
            draw.line(
                [(line_x0, line_y0), (line_x0 + cell_size, line_y0)],
                fill="black",
                width=5,
            )

    img.save(filename, "PNG")


def generate_chain_of_draft_str(puzzle_data: Puzzle) -> str:
    """Generates the concise, shorthand-style Chain-of-Draft string."""
    m, n = puzzle_data["grid_size"]
    grid = puzzle_data["grid"]

    drafts = []
    drafts.append(f"{m}x{n}")

    waypoints = {
        grid[r][c]: (r, c) for r in range(m) for c in range(n) if grid[r][c] > 0
    }
    if waypoints:
        wp_str = ", ".join([f"{k}@{waypoints[k]}" for k in sorted(waypoints.keys())])
        drafts.append(wp_str)

    blocked_cells = sorted(list(puzzle_data.get("blocked_cells", set())))
    if blocked_cells:
        b_str = ", ".join([str(pos) for pos in blocked_cells])
        drafts.append(f"B@{b_str}")

    walls = sorted(list(puzzle_data.get("walls", set())))
    if walls:
        # Format walls like W@((r1,c1)-(r2,c2)), ...
        w_str = ", ".join([f"({p1}-{p2})" for p1, p2 in walls])
        drafts.append(f"W@{w_str}")

    return "\n".join(drafts)


def worker_generate_cod_item(task_info: dict[str, Any]) -> bool:
    """Worker function to generate a single puzzle image and a CoD label file."""
    puzzle_num = task_info["puzzle_num"]
    output_dir = Path(task_info["output_dir"])
    map_size = task_info["map_size"]

    try:
        # 1. Generate puzzle data
        result = generate_puzzle(m=map_size[0], n=map_size[1], has_walls=True)
        if not result:
            return False
        puzzle_data, _ = result

        # 2. Save the puzzle image
        images_dir = output_dir / "images"
        image_filename = images_dir / f"puzzle_{puzzle_num:03d}.png"
        save_puzzle_as_png(puzzle_data, str(image_filename))

        # 3. Generate CoD components
        chain_of_draft_str = generate_chain_of_draft_str(puzzle_data)

        m, n = puzzle_data["grid_size"]
        waypoints_dict = {
            str(puzzle_data["grid"][r][c]): [r, c]
            for r in range(m)
            for c in range(n)
            if puzzle_data["grid"][r][c] > 0
        }

        final_ans = {
            "m": m,
            "n": n,
            "waypoints": waypoints_dict,
            "blocked_cells": sorted(
                [list(pos) for pos in puzzle_data.get("blocked_cells", set())]
            ),
            "walls": sorted(
                [[list(p1), list(p2)] for p1, p2 in puzzle_data.get("walls", set())]
            ),
        }

        # 4. Create and save the final CoD JSON label
        labels_dir = output_dir / "labels"
        label_data = {
            "chain_of_draft": chain_of_draft_str,
            "final_ans": final_ans,
        }
        label_filename = labels_dir / f"puzzle_{puzzle_num:03d}.json"
        with open(label_filename, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2)

        return True
    except Exception as e:
        print(f"[Worker {{puzzle_num}}] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_cod_dataset(num_puzzles: int, map_size: tuple[int, int]):
    """Generates a CoD dataset in parallel using a multiprocessing pool."""
    dataset_name = f"cod_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = PROJECT_ROOT / "zip_puzzles" / dataset_name
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels").mkdir(parents=True, exist_ok=True)

    print(
        f"Starting Chain-of-Draft (CoD) dataset generation for {num_puzzles} puzzles of size {map_size}..."
    )
    print(f"Dataset will be saved in: {dataset_dir}")

    tasks = [
        {
            "puzzle_num": i + 1,
            "output_dir": str(dataset_dir),
            "map_size": map_size,
        }
        for i in range(num_puzzles)
    ]

    num_processes = max(1, int(cpu_count() * 0.8))
    print(f"Using {num_processes} processes to generate dataset...")

    try:
        from tqdm import tqdm

        with Pool(processes=num_processes) as pool:
            results = list(
                tqdm(pool.imap(worker_generate_cod_item, tasks), total=num_puzzles)
            )
    except ImportError:
        print("tqdm not found, running without a progress bar.")
        with Pool(processes=num_processes) as pool:
            results = pool.map(worker_generate_cod_item, tasks)

    success_count = sum(1 for r in results if r is True)
    print(f"\nSuccessfully generated {success_count}/{num_puzzles} puzzles.")
    print(f"Dataset located at: {dataset_dir}")


if __name__ == "__main__":
    NUM_PUZZLES_TO_GENERATE = 5
    MAP_SIZE = (6, 6)
    create_cod_dataset(NUM_PUZZLES_TO_GENERATE, MAP_SIZE)
