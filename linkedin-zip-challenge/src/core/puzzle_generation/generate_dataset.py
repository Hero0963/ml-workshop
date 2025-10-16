# src/core/puzzle_generation/generate_dataset.py

import os
import random
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from loguru import logger

# --- Configuration ---
NUM_TO_GENERATE = 6
TIMEOUT_PER_PUZZLE_ATTEMPT = 20  # Max time for a single pathfinding attempt

# Grid size parameters
GRID_SIZE_MIN = 5
GRID_SIZE_MAX = 7

# Obstacle parameters
MAX_BLOCKED_CELLS_TO_ATTEMPT = 3


def worker_generate_and_save(task_params: dict) -> dict | None:
    """
    A worker function designed to be run in a separate process.
    It generates a single puzzle, saves its GIF, and returns its data.
    This function is self-contained with its own imports.
    """
    # --- Self-contained Imports ---
    import pprint
    from pathlib import Path
    from loguru import logger
    from src.core.puzzle_generation.puzzle_generator import generate_puzzle
    from src.core.utils import save_animation_as_gif

    task_id = task_params["task_id"]
    output_dir = Path(task_params["output_dir"])
    gen_params = task_params["gen_params"]

    try:
        logger.info(f"[Task {task_id}] Starting generation with params: {gen_params}")

        result = generate_puzzle(**gen_params)

        if not result:
            logger.error(
                f"[Task {task_id}] Failed to generate puzzle even after all retries. Returning None."
            )
            return None

        puzzle, solution_path = result

        # --- Save GIF ---
        gif_dir = output_dir / "gifs"
        gif_filename = gif_dir / f"puzzle_{task_id:02d}.gif"
        logger.info(f"[Task {task_id}] Saving visualization to {gif_filename}...")
        try:
            save_animation_as_gif(puzzle, solution_path, str(gif_filename))
        except Exception:
            logger.exception(
                f"[Task {task_id}] Could not save GIF due to an unexpected error."
            )

        # --- Format Data for Python File ---
        puzzle_name = f"puzzle_{task_id:02d}"
        layout_str = pprint.pformat(puzzle["puzzle_layout"], indent=4)
        walls_str = pprint.pformat(puzzle["walls"], indent=4)
        solution_str = pprint.pformat(solution_path, indent=4)

        data_string = (
            f"# --- Puzzle {task_id:02d} Data ---\n"
            f"{puzzle_name}_layout = {layout_str}\n\n"
            f"{puzzle_name}_data = parse_puzzle_layout({puzzle_name}_layout)\n"
            f'{puzzle_name}_data["walls"] = {walls_str}\n'
            f"solution_{task_id:02d} = {solution_str}\n"
        )

        return {
            "task_id": task_id,
            "data_string": data_string,
            "puzzle_name": puzzle_name,
        }

    except Exception:
        logger.exception(f"[Task {task_id}] An unexpected error occurred in worker.")
        return None


def main():
    """Main function to orchestrate the dataset generation."""
    # 1. Setup Directories and Logging
    main_output_dir = Path("puzzle_dataset")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = main_output_dir / f"dataset_{timestamp}"
    gif_dir = output_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "generation.log"
    logger.add(log_file, level="INFO")
    logger.info(f"Output will be saved in: {output_dir.resolve()}")

    # # 2. Create Task Definitions
    # tasks = []
    # for i in range(1, NUM_TO_GENERATE + 1):
    #     m = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    #     n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    #     tasks.append(
    #         {
    #             "task_id": i,
    #             "output_dir": str(output_dir),
    #             "gen_params": {
    #                 "m": m,
    #                 "n": n,
    #                 "has_walls": random.choice([True, False]),
    #                 "num_blocked_cells": random.randint(
    #                     0, MAX_BLOCKED_CELLS_TO_ATTEMPT
    #                 ),
    #                 "timeout_per_attempt": TIMEOUT_PER_PUZZLE_ATTEMPT,
    #             },
    #         }
    #     )

    # 2. Create Task Definitions simple version
    tasks = []
    for i in range(1, NUM_TO_GENERATE + 1):
        size = 6
        m = size
        n = size
        tasks.append(
            {
                "task_id": i,
                "output_dir": str(output_dir),
                "gen_params": {
                    "m": m,
                    "n": n,
                    "has_walls": random.choice([True, False]),
                    "num_blocked_cells": 0,
                    "timeout_per_attempt": TIMEOUT_PER_PUZZLE_ATTEMPT,
                },
            }
        )

    # 3. Execute tasks in parallel using imap_unordered
    logger.info(f"Starting parallel generation of {len(tasks)} puzzles...")
    results = []
    num_processes = max(1, int(os.cpu_count() * 0.75))
    logger.info(
        f"Creating a pool with {num_processes} processes (75% of available cores)."
    )
    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_generate_and_save, tasks)

        for result in results_iterator:
            if result:
                results.append(result)
                logger.success(
                    f"Result for task {result['task_id']} received and processed."
                )
            else:
                logger.error("A worker task failed to generate a puzzle.")

    # 4. Aggregate successful results into a single file
    if not results:
        logger.error("No puzzles were generated successfully. Aborting file write.")
        return

    results.sort(key=lambda r: r["task_id"])

    output_data_file = output_dir / "puzzles.py"
    with open(output_data_file, "w", encoding="utf-8") as f:
        f.write("# Auto-generated puzzle dataset\n")
        f.write("from src.core.utils import parse_puzzle_layout\n\n")

        for res in results:
            f.write(res["data_string"])
            f.write("\n")

        f.write("# --- Test Suite ---\n")
        f.write("puzzles_to_test = [\n")
        for res in results:
            f.write(
                f"    ({res['puzzle_name']}_data, solution_{res['task_id']:02d}, \"{res['puzzle_name']}\"),\n"
            )
        f.write("]\n")

    logger.success(
        f"Dataset generation complete. {len(results)}/{NUM_TO_GENERATE} puzzles saved."
    )
    logger.info(f"Data file: {output_data_file}")
    logger.info(f"GIFs saved in: {gif_dir}")


if __name__ == "__main__":
    main()
