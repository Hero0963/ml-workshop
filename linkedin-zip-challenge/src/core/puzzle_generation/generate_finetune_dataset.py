# src/core/puzzle_generation/generate_finetune_dataset.py

import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from PIL import Image, ImageDraw, ImageFont
    from src.core.puzzle_generation.puzzle_generator import generate_puzzle
    from src.core.utils import Puzzle  # Use the TypedDict for structure
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print("Please ensure you are in the correct environment and all paths are correct.")
    exit(1)


def save_puzzle_as_png(
    puzzle_data: Puzzle,
    filename: str = "puzzle.png",
) -> None:
    """Generates and saves a static image of the unsolved puzzle."""
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
        if r1 == r2:
            line_x0 = margin + max(c1, c2) * cell_size
            line_y0 = margin + r1 * cell_size
            draw.line(
                [(line_x0, line_y0), (line_x0, line_y0 + cell_size)],
                fill="black",
                width=5,
            )
        else:
            line_x0 = margin + c1 * cell_size
            line_y0 = margin + max(r1, r2) * cell_size
            draw.line(
                [(line_x0, line_y0), (line_x0 + cell_size, line_y0)],
                fill="black",
                width=5,
            )

    img.save(filename, "PNG")


def worker_generate_item(task_info: dict) -> bool:
    """Worker function to generate a single puzzle image and label file."""
    puzzle_num = task_info["puzzle_num"]
    output_dir = Path(task_info["output_dir"])
    map_size = task_info["map_size"]

    try:
        # 1. Generate puzzle data
        result = generate_puzzle(m=map_size[0], n=map_size[1], has_walls=True)
        if not result:
            print(f"[Worker {puzzle_num}] Puzzle generation failed, skipping.")
            return False
        puzzle_data, _ = result

        # 2. Save the puzzle image
        images_dir = output_dir / "images"
        image_filename = images_dir / f"puzzle_{puzzle_num:03d}.png"
        save_puzzle_as_png(puzzle_data, str(image_filename))

        # 3. Create and save the JSON label
        labels_dir = output_dir / "labels"
        label_data = {
            "layout": puzzle_data["puzzle_layout"],
            "walls": [
                {"cell1": list(c1), "cell2": list(c2)}
                for c1, c2 in sorted(list(puzzle_data["walls"]))
            ],
        }
        label_filename = labels_dir / f"puzzle_{puzzle_num:03d}.json"
        with open(label_filename, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2)

        return True
    except Exception as e:
        print(f"[Worker {puzzle_num}] An unexpected error occurred: {e}")
        return False


def create_finetune_dataset(num_puzzles: int, map_size: tuple[int, int]):
    """Generates a dataset in parallel using a multiprocessing pool."""
    print(
        f"Starting dataset generation for {num_puzzles} puzzles of size {map_size}..."
    )

    dataset_name = f"finetune_dataset_{time.strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = PROJECT_ROOT / "zip_puzzles" / dataset_name
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Dataset will be saved in: {dataset_dir}")

    tasks = [
        {
            "puzzle_num": i + 1,
            "output_dir": str(dataset_dir),
            "map_size": map_size,
        }
        for i in range(num_puzzles)
    ]

    num_processes = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_processes} processes to generate dataset...")

    with Pool(processes=num_processes) as pool:
        results = pool.map(worker_generate_item, tasks)

    success_count = sum(1 for r in results if r is True)
    print(f"\nSuccessfully generated {success_count}/{num_puzzles} puzzles.")


if __name__ == "__main__":
    NUM_PUZZLES_TO_GENERATE = 200
    MAP_SIZE = (6, 6)
    create_finetune_dataset(NUM_PUZZLES_TO_GENERATE, MAP_SIZE)
