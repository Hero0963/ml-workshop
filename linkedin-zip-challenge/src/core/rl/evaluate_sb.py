# src/core/rl/evaluate_sb.py
"""
Script to evaluate a trained Stable-Baselines3 agent.

This script loads a trained model (.zip), runs it on a puzzle,
and saves the resulting path as a GIF for visualization.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import torch
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3 import DQN

from src.core.puzzle_generator import generate_puzzle
from src.core.rl.rl_env import PuzzleEnv
from src.core.utils import Puzzle, save_animation_as_gif

# --- Add project root to sys.path for absolute imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---


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
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info(f"Starting evaluation with SB3 model: {model_path}")

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 2. Get the puzzle for evaluation ---
    if use_training_puzzle:
        # Use the exact puzzle from the overfitting training
        dataset_path = Path(
            "D:/it_project/github_sync/ml-workshop/linkedin-zip-challenge/datasets/rl_datasets/rl_dataset_2025-10-12_092521/puzzles.pkl"
        )
        puzzles = load_puzzles_from_dataset(dataset_path)
        puzzle = puzzles[0]
        logger.info("Evaluating on the first puzzle from the training dataset.")
    else:
        # Generate a new, random puzzle
        logger.info(f"Generating a new {puzzle_size}x{puzzle_size} puzzle...")
        result = generate_puzzle(
            m=puzzle_size, n=puzzle_size, num_blocked_cells=num_obstacles
        )
        if not result:
            logger.error("Failed to generate a puzzle for evaluation.")
            return
        puzzle, _ = result

    # --- 3. Load Agent and Environment ---
    # The model needs the wrapped env, but we need the raw env for visualization info
    # Pass a generous step limit to the environment constructor
    max_eval_steps = (
        puzzle["grid_size"][0] * puzzle["grid_size"][1] * 8
    )  # Increased to 8x for safety
    raw_env = PuzzleEnv(puzzle, max_steps=max_eval_steps)
    wrapped_env = FlattenObservation(raw_env)

    model = DQN.load(model_path, device=device)
    logger.success("Model and evaluation environment are ready.")

    # --- 4. Run Inference Loop ---
    obs, _ = wrapped_env.reset()
    path_taken = [raw_env.waypoints[0]]  # Start with the first waypoint
    total_reward = 0
    done = False
    step_count = 0

    # We still use max_eval_steps here as a safeguard against potential infinite loops.
    while not done and step_count < max_eval_steps:
        step_count += 1
        # Use deterministic=True for pure exploitation
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = wrapped_env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Get the agent's location from the underlying raw environment
        path_taken.append(tuple(raw_env._agent_location))

    logger.info(f"Evaluation finished in {step_count} steps.")
    logger.info(f"Total reward: {total_reward:.2f}")

    # --- 5. Visualize and Save Result ---
    if terminated:
        logger.success("Agent successfully completed the puzzle!")
    else:
        logger.warning("Agent did not complete the puzzle within the step limit.")

    save_animation_as_gif(puzzle, path_taken, str(output_gif_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Stable-Baselines3 Agent."
    )
    default_model_path = (
        PROJECT_ROOT / "linkedin-zip-challenge" / "models" / "dqn_sb_single.zip"
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
        default="evaluation_sb.gif",
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
