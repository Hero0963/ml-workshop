# src/core/rl/generate_rl_dataset.py
"""
Generates a dataset of puzzles specifically for RL training.

This script creates a timestamped directory and outputs two files:
1. puzzles_details.log: A human-readable log of all generated puzzles.
2. puzzles.pkl: A pickle file containing the list of Puzzle objects for the
   training script to consume.

It leverages multiprocessing to speed up generation.
"""

import os
import pickle
import pprint
import random
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from tqdm import tqdm

# --- Add project root to sys.path for absolute imports ---
# This is a common pattern to make imports work when running a script from a sub-directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---

from src.core.puzzle_generation.puzzle_generator import generate_puzzle  # noqa: E402
from src.core.utils import Puzzle  # noqa: E402

# --- Configuration ---
PUZZLE_CONFIG: List[Dict[str, int]] = [
    {"size": 6, "count": 100},
    {"size": 7, "count": 100},
]
NUM_BLOCKED_CELLS = 0
BASE_OUTPUT_DIR = PROJECT_ROOT / "linkedin-zip-challenge" / "datasets" / "rl_datasets"
TIMEOUT_PER_PUZZLE_ATTEMPT = 20  # Max time for a single pathfinding attempt


def worker_generate_puzzle(task_params: Dict[str, Any]) -> Puzzle | None:
    """
    A worker function for multiprocessing. Generates a single puzzle.
    Self-contained to avoid issues with pickling complex objects.
    """
    from loguru import logger

    task_id = task_params["task_id"]
    gen_params = task_params["gen_params"]

    try:
        result = generate_puzzle(**gen_params)
        if result:
            puzzle, _ = result
            logger.success(
                f"[Task {task_id}] Successfully generated a {gen_params['m']}x{gen_params['n']} puzzle."
            )
            return puzzle
        else:
            logger.error(f"[Task {task_id}] Failed to generate puzzle.")
            return None
    except Exception as e:
        logger.exception(f"[Task {task_id}] An unexpected error occurred: {e}")
        return None


def save_human_readable_log(puzzles: List[Puzzle], output_dir: Path):
    """Saves the puzzle details in a human-readable text file."""
    log_path = output_dir / "puzzles_details.log"
    logger.info(f"Saving human-readable puzzle details to {log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        for i, puzzle in enumerate(puzzles):
            f.write(f"--- Puzzle {i+1} ---\\n")
            # Use pprint to format the dictionary nicely
            f.write(pprint.pformat(puzzle, indent=2, width=120))
            f.write("\\n\\n")
    logger.success("Human-readable log saved.")


def save_pickle_file(puzzles: List[Puzzle], output_dir: Path):
    """Saves the list of puzzle objects to a pickle file."""
    pickle_path = output_dir / "puzzles.pkl"
    logger.info(f"Saving puzzle objects to pickle file: {pickle_path}")
    with open(pickle_path, "wb") as f:
        pickle.dump(puzzles, f)
    logger.success("Pickle file saved.")


def main():
    """Main function to orchestrate the dataset generation."""
    # 1. Setup Directories and Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = BASE_OUTPUT_DIR / f"rl_dataset_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "generation.log"
    logger.add(log_file, level="INFO")
    logger.info(f"Output will be saved in: {output_dir.resolve()}")

    # 2. Create Task Definitions
    tasks: List[Dict[str, Any]] = []
    task_id_counter = 1
    for config in PUZZLE_CONFIG:
        size = config["size"]
        count = config["count"]
        for _ in range(count):
            tasks.append(
                {
                    "task_id": task_id_counter,
                    "gen_params": {
                        "m": size,
                        "n": size,
                        "has_walls": random.choice([True, False]),
                        "num_blocked_cells": NUM_BLOCKED_CELLS,
                        "timeout_per_attempt": TIMEOUT_PER_PUZZLE_ATTEMPT,
                    },
                }
            )
            task_id_counter += 1

    # 3. Execute tasks in parallel
    total_tasks = len(tasks)
    logger.info(f"Starting parallel generation of {total_tasks} puzzles...")
    puzzles: List[Puzzle] = []
    with Pool(processes=os.cpu_count()) as pool, tqdm(total=total_tasks) as pbar:
        for result in pool.imap_unordered(worker_generate_puzzle, tasks):
            if result:
                puzzles.append(result)
            pbar.update(1)

    # 4. Save the collected puzzles
    if not puzzles:
        logger.error("No puzzles were generated successfully. Aborting file write.")
        return

    logger.success(f"Successfully generated {len(puzzles)}/{total_tasks} puzzles.")

    save_human_readable_log(puzzles, output_dir)
    save_pickle_file(puzzles, output_dir)

    logger.info("--- Dataset Generation Complete ---")
    logger.info(f"Human-readable log: {output_dir / 'puzzles_details.log'}")
    logger.info(f"Pickle dataset file: {output_dir / 'puzzles.pkl'}")


if __name__ == "__main__":
    main()
