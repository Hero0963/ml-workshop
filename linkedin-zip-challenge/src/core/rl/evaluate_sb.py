# src/core/rl/evaluate_sb.py
"""
Script to evaluate a trained Stable-Baselines3 agent.

This script loads a trained model (.zip), runs it on a puzzle,
and saves the resulting path as a GIF for visualization.

Enhanced version for detailed analysis:
- Creates a detailed GIF showing current position and revisited paths.
- Logs the exact path to a separate file.
- Adapted for CNN-based models.
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import List
from datetime import datetime

import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.rl_env import PuzzleEnv
from src.core.utils import Puzzle

# --- Add project root to sys.path for absolute imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---


def save_detailed_animation_as_gif(
    puzzle: Puzzle,
    solution_path: list[tuple[int, int]] | None,
    filename: str = "solution.gif",
    speed: int = 250,
) -> None:
    """Generates and saves a detailed solution animation as a GIF file."""
    if not solution_path:
        logger.warning("No solution path to generate GIF.")
        return

    grid: list[list[int]] = puzzle["grid"]
    walls: set[tuple[tuple[int, int], tuple[int, int]]] = puzzle.get("walls", set())
    blocked_cells: set[tuple[int, int]] = puzzle.get("blocked_cells", set())
    height, width = puzzle["grid_size"]
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
        partial_path = solution_path[:i]
        path_counts = Counter(partial_path)

        for r in range(height):
            for c in range(width):
                x0 = margin + c * cell_size
                y0 = margin + r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                rect = [(x0, y0), (x1, y1)]
                pos = (r, c)

                fill_color = "white"
                if pos in path_counts:
                    if path_counts[pos] > 1:
                        fill_color = "#FFC0CB"  # Pink for revisited
                    else:
                        fill_color = "#cccccc"  # Grey for path

                # Current head of the path
                if i > 0 and pos == solution_path[i - 1]:
                    fill_color = "#90EE90"  # Light green for current head

                draw.rectangle(rect, fill=fill_color)

                if pos in blocked_cells:
                    draw.rectangle(rect, fill="black")
                    continue

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
            if r1 == r2:
                line_x0 = margin + max(c1, c2) * cell_size
                line_y0 = margin + r1 * cell_size
                line_x1 = line_x0
                line_y1 = line_y0 + cell_size
                draw.line(
                    [(line_x0, line_y0), (line_x1, line_y1)], fill="black", width=3
                )
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
    logger.info(f"Detailed animation saved to {filename}")


def load_puzzles_from_dataset(dataset_path: Path) -> List[Puzzle]:
    """Loads a list of puzzles from a pickle file."""
    if not dataset_path.exists():
        logger.error(f"Dataset file not found at: {dataset_path}")
        sys.exit(1)
    with open(dataset_path, "rb") as f:
        puzzles = pickle.load(f)
    return puzzles


def run_evaluation(
    model_path: Path,
    puzzle_size: int,
    num_obstacles: int,
    output_gif_path: Path,
    use_training_puzzle: bool,
):
    """Loads a model and runs it on a puzzle."""
    # --- 1. Setup ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_ROOT / "linkedin-zip-challenge" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path_log_file = log_dir / f"evaluation_path_{timestamp}.log"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(
        path_log_file, format="{message}", level="SUCCESS"
    )  # Special logger for path

    logger.info(f"Starting evaluation with SB3 model: {model_path}")

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 2. Get the puzzle for evaluation ---
    if use_training_puzzle:
        dataset_path = Path(
            "D:/it_project/github_sync/ml-workshop/linkedin-zip-challenge/datasets/rl_datasets/rl_dataset_2025-10-12_092521/puzzles.pkl"
        )
        puzzles = load_puzzles_from_dataset(dataset_path)
        puzzle = puzzles[0]
        logger.info("Evaluating on the first puzzle from the training dataset.")
    else:
        logger.info(f"Generating a new {puzzle_size}x{puzzle_size} puzzle...")
        result = generate_puzzle(
            m=puzzle_size, n=puzzle_size, num_blocked_cells=num_obstacles
        )
        if not result:
            logger.error("Failed to generate a puzzle for evaluation.")
            return
        puzzle, _ = result

    # --- 3. Load Agent and Environment ---
    max_eval_steps = puzzle["grid_size"][0] * puzzle["grid_size"][1] * 2
    env = PuzzleEnv(puzzle, max_steps=max_eval_steps)

    # No longer need FlattenObservation for our CustomCnn policy
    model = DQN.load(model_path, env=env, device=device)
    logger.success("Model and evaluation environment are ready.")

    # --- 4. Run Inference Loop ---
    obs, _ = env.reset()
    path_taken = [env.start_pos]
    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < max_eval_steps:
        step_count += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        path_taken.append(tuple(env._agent_location))

    logger.info(f"Evaluation finished in {step_count} steps.")
    logger.info(f"Total reward: {total_reward:.2f}")

    # --- 5. Log, Visualize and Save Result ---
    if terminated:
        logger.success("Agent successfully completed the puzzle!")
    else:
        logger.warning("Agent did not complete the puzzle within the step limit.")

    # Log the path to a separate file
    logger.success(f"Saving evaluation path to {path_log_file}")
    for pos in path_taken:
        logger.log("SUCCESS", str(pos))

    # Save the new detailed GIF
    save_detailed_animation_as_gif(puzzle, path_taken, str(output_gif_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Stable-Baselines3 Agent."
    )
    # Default to the new CNN model
    default_model_path = (
        PROJECT_ROOT / "linkedin-zip-challenge" / "models" / "dqn_sb_cnn_single.zip"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_model_path),
        help="Path to the trained SB3 model (.zip file).",
    )
    parser.add_argument(
        "--size", type=int, default=6, help="The size (m=n) of the puzzle to generate."
    )
    parser.add_argument(
        "--obstacles",
        type=int,
        default=0,
        help="Number of obstacles in the generated puzzle.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_sb_detailed.gif",  # New default name
        help="Path to save the output GIF file.",
    )
    parser.add_argument(
        "--use_training_puzzle",
        action="store_true",
        help="If set, evaluates on the first puzzle from the training dataset.",
    )
    args = parser.parse_args()

    run_evaluation(
        model_path=Path(args.model_path),
        puzzle_size=args.size,
        num_obstacles=args.obstacles,
        output_gif_path=Path(args.output),
        use_training_puzzle=args.use_training_puzzle,
    )
