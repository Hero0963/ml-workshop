# src/core/rl/evaluate.py
"""
Script to evaluate a trained DQN agent.

This script loads a trained model, runs it on a new, unseen puzzle,
and saves the resulting path as a GIF for visualization.
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.dqn_agent import DQNAgent
from src.core.rl.rl_env import PuzzleEnv
from src.core.utils import save_animation_as_gif


# --- Add project root to sys.path for absolute imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---


def run_evaluation(
    model_path: Path,
    puzzle_size: int = 6,
    num_obstacles: int = 0,
    output_gif_path: Path = Path("evaluation.gif"),
):
    """Loads a model and runs it on a new puzzle."""
    # --- 1. Setup ---
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info(f"Starting evaluation with model: {model_path}")

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 2. Generate a new puzzle for evaluation ---
    logger.info(f"Generating a new {puzzle_size}x{puzzle_size} puzzle...")
    result = generate_puzzle(
        m=puzzle_size, n=puzzle_size, num_blocked_cells=num_obstacles
    )
    if not result:
        logger.error("Failed to generate a puzzle for evaluation.")
        return
    puzzle, _ = result
    env = PuzzleEnv(puzzle)

    # --- 3. Load Agent and Model ---
    obs_shape = 4  # agent_loc (2) + waypoint_loc (2)
    n_actions = env.action_space.n
    agent = DQNAgent(obs_shape=obs_shape, n_actions=n_actions, device=device)

    state_dict = torch.load(model_path, map_location=device)
    agent.online_net.load_state_dict(state_dict)
    agent.online_net.eval()  # Set the network to evaluation mode

    logger.success("Model loaded and evaluation environment is ready.")

    # --- 4. Run Inference Loop ---
    obs, _ = env.reset()
    path_taken = [env.waypoints[0]]
    total_reward = 0
    done = False
    step_count = 0
    max_eval_steps = puzzle_size * puzzle_size * 2

    while not done and step_count < max_eval_steps:
        step_count += 1
        processed_obs = agent.preprocess_obs(obs)
        # Use epsilon = 0 for pure exploitation
        action = agent.select_action(processed_obs, epsilon=0)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        path_taken.append(tuple(obs["agent_location"]))

    logger.info(f"Evaluation finished in {step_count} steps.")
    logger.info(f"Total reward: {total_reward:.2f}")

    # --- 5. Visualize and Save Result ---
    if terminated:
        logger.success("Agent successfully completed the puzzle!")
    else:
        logger.warning("Agent did not complete the puzzle within the step limit.")

    save_animation_as_gif(puzzle, path_taken, str(output_gif_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN Agent.")
    default_model_path = (
        PROJECT_ROOT / "linkedin-zip-challenge" / "models" / "dqn_agent_best.pth"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_model_path),
        help="Path to the trained model state_dict file.",
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
        default="evaluation.gif",
        help="Path to save the output GIF file.",
    )
    args = parser.parse_args()

    run_evaluation(
        model_path=Path(args.model_path),
        puzzle_size=args.size,
        num_obstacles=args.obstacles,
        output_gif_path=Path(args.output),
    )
