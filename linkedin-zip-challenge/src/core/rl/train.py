# src\core\rl\train.py
"""
Main training script for the DQN Agent.

This script orchestrates the training process by:
1. Loading a pre-generated set of training puzzles.
2. Initializing the environment and the DQN agent.
3. Running the main training loop, where the agent interacts with the
   environment, stores experiences, and learns.
4. Logging progress and saving the trained model.

Requires:
- torch
- numpy
- loguru
- tqdm
- gymnasium
"""

import random
import pickle
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.core.rl.dqn_agent import DQNAgent
from src.core.rl.rl_env import PuzzleEnv
from src.core.utils import Puzzle

# --- Add project root to sys.path for absolute imports ---
# This is a common pattern to make imports work when running a script from a sub-directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---


# --- Configuration ---
CONFIG: Dict[str, any] = {
    # --- IMPORTANT: Please update this path to your generated .pkl file ---
    "DATASET_PATH": "D:/it_project/github_sync/ml-workshop/linkedin-zip-challenge/datasets/rl_datasets/rl_dataset_2025-10-12_092521/puzzles.pkl",
    # Training settings
    "TOTAL_TIMESTEPS": 200_000,
    "LEARNING_RATE": 5e-4,  # A slightly higher learning rate can be effective
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.05,
    "EPSILON_DECAY": 50_000,  # Slower decay to encourage more exploration
    # Replay Buffer & Agent settings
    "BATCH_SIZE": 128,  # Larger batch size for more stable gradients
    "REPLAY_CAPACITY": 100_000,  # Larger buffer to hold more diverse experiences
    "TARGET_UPDATE_FREQ": 1000,  # Steps
    "AGENT_TAU": 0.995,  # Soft update factor
    # Logging and Saving
    "MODEL_SAVE_PATH": PROJECT_ROOT
    / "linkedin-zip-challenge"
    / "models"
    / "dqn_agent.pth",
}


def load_puzzles(dataset_path: Path) -> List[Puzzle]:
    """Loads a list of puzzles from a pickle file."""
    if not dataset_path.exists():
        logger.error(f"Dataset file not found at: {dataset_path}")
        logger.error(
            "Please run `generate_rl_dataset.py` first or update the DATASET_PATH in the config."
        )
        sys.exit(1)

    logger.info(f"Loading puzzle dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        puzzles = pickle.load(f)
    logger.success(f"Loaded {len(puzzles)} puzzles.")
    return puzzles


def main():
    """Main function to run the training loop."""
    # --- 1. Setup ---
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.info("Starting DQN training process...")
    logger.info(f"Configuration: {CONFIG}")

    CONFIG["MODEL_SAVE_PATH"].parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 2. Puzzle and Environment Initialization ---
    dataset_path = Path(CONFIG["DATASET_PATH"])
    training_puzzles = load_puzzles(dataset_path)
    env = PuzzleEnv(random.choice(training_puzzles))

    # --- 3. Agent Initialization ---
    obs_shape = 4  # agent_loc (2) + waypoint_loc (2)
    n_actions = env.action_space.n
    agent = DQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        device=device,
        learning_rate=CONFIG["LEARNING_RATE"],
        gamma=CONFIG["GAMMA"],
        replay_capacity=CONFIG["REPLAY_CAPACITY"],
        batch_size=CONFIG["BATCH_SIZE"],
        target_update_freq=CONFIG["TARGET_UPDATE_FREQ"],
        tau=CONFIG["AGENT_TAU"],
    )

    # --- 4. Training Loop ---
    obs, _ = env.reset()
    processed_obs = agent.preprocess_obs(obs)
    episode_reward = 0
    episode_rewards: List[float] = []

    for step in tqdm(range(1, CONFIG["TOTAL_TIMESTEPS"] + 1), desc="Training Steps"):
        epsilon = np.interp(
            step,
            [0, CONFIG["EPSILON_DECAY"]],
            [CONFIG["EPSILON_START"], CONFIG["EPSILON_END"]],
        )

        action = agent.select_action(processed_obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        processed_next_obs = agent.preprocess_obs(next_obs)

        done = terminated or truncated
        agent.buffer.push(processed_obs, action, reward, processed_next_obs, done)

        processed_obs = processed_next_obs
        episode_reward += reward

        # Start learning once the buffer has enough samples
        if len(agent.buffer) > CONFIG["BATCH_SIZE"]:
            agent.learn()

        if done:
            episode_rewards.append(episode_reward)
            if len(episode_rewards) % 10 == 0:  # Log every 10 episodes
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(
                    f"Step: {step}, Episodes: {len(episode_rewards)}, "
                    f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}"
                )

            # Reset for next episode with a new random puzzle
            env = PuzzleEnv(random.choice(training_puzzles))
            obs, _ = env.reset()
            processed_obs = agent.preprocess_obs(obs)
            episode_reward = 0

    # --- 5. Save Model ---
    logger.success("Training finished.")
    torch.save(agent.online_net.state_dict(), CONFIG["MODEL_SAVE_PATH"])
    logger.info(f"Model saved to {CONFIG['MODEL_SAVE_PATH']}")


if __name__ == "__main__":
    main()
