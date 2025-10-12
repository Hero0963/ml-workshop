# src/core/rl/train_single_sb.py
"""
Training script for a single puzzle using the Stable-Baselines3 library.

This script is for overfitting on one puzzle to test the environment and
reward function, leveraging the robust DQN implementation from SB3.

Key Features:
1. Uses `stable_baselines3.DQN`.
2. Uses `gymnasium.wrappers.FlattenObservation` to make the env compatible.
3. Training loop is simplified to `model.learn()`.
4. Model save paths are unique to this script to avoid conflicts.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation


# --- Add project root to sys.path for absolute imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---

from src.core.rl.rl_env import PuzzleEnv  # noqa: E402
from src.core.utils import Puzzle  # noqa: E402

# --- Configuration ---
CONFIG: Dict[str, any] = {
    # --- Path Settings ---
    "DATASET_PATH": "D:/it_project/github_sync/ml-workshop/linkedin-zip-challenge/datasets/rl_datasets/rl_dataset_2025-10-12_092521/puzzles.pkl",
    "MODEL_SAVE_PATH": "models/dqn_sb_single.zip",  # SB3 saves models in a .zip file
    # --- Training Settings ---
    "TOTAL_TIMESTEPS": 200_000,
    "LEARNING_RATE": 5e-4,
    "GAMMA": 0.99,
    "EPSILON_DECAY_STEPS": 50_000,
    # --- Agent & Buffer Settings ---
    "BATCH_SIZE": 128,
    "REPLAY_CAPACITY": 100_000,
    "TARGET_UPDATE_FREQ": 1000,  # In steps
    "AGENT_TAU": 0.995,
}


def load_puzzles(dataset_path: Path) -> List[Puzzle]:
    """Loads a list of puzzles from a pickle file."""
    if not dataset_path.exists():
        logger.error(f"Dataset file not found at: {dataset_path}")
        sys.exit(1)

    logger.info(f"Loading puzzle dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        puzzles = pickle.load(f)
    logger.success(f"Loaded {len(puzzles)} puzzles.")
    return puzzles


def main():
    """Main function to run the training loop."""
    # --- 1. Setup ---
    log_dir = PROJECT_ROOT / "linkedin-zip-challenge" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "training_sb_single_{time:YYYY-MM-DD_HH-mm-ss}.log"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file_path, level="INFO")

    logger.info("Starting Stable-Baselines3 single-puzzle overfitting test...")

    base_model_dir = PROJECT_ROOT / "linkedin-zip-challenge"
    save_path = base_model_dir / CONFIG["MODEL_SAVE_PATH"]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- 2. Puzzle and Environment Initialization ---
    dataset_path = Path(CONFIG["DATASET_PATH"])
    training_puzzles = load_puzzles(dataset_path)
    fixed_puzzle = training_puzzles[0]

    # SB3 works best with a vectorized environment, even for a single instance.
    # We also need to wrap our custom env to flatten the Dict observation space.
    def make_env():
        env = PuzzleEnv(fixed_puzzle)
        env = FlattenObservation(env)
        return env

    env = make_vec_env(make_env, n_envs=1)
    logger.info("Created and wrapped environment for Stable-Baselines3.")

    # --- 3. Agent Initialization ---
    exploration_fraction = CONFIG["EPSILON_DECAY_STEPS"] / CONFIG["TOTAL_TIMESTEPS"]

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=CONFIG["LEARNING_RATE"],
        buffer_size=CONFIG["REPLAY_CAPACITY"],
        learning_starts=CONFIG[
            "BATCH_SIZE"
        ],  # Start learning after one batch is filled
        batch_size=CONFIG["BATCH_SIZE"],
        tau=CONFIG["AGENT_TAU"],
        gamma=CONFIG["GAMMA"],
        train_freq=(1, "step"),  # Train every step
        target_update_interval=CONFIG["TARGET_UPDATE_FREQ"],
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,  # Corresponds to EPSILON_START
        exploration_final_eps=0.05,  # Corresponds to EPSILON_END
        verbose=1,  # Set to 1 to see SB3 training logs
        tensorboard_log=str(log_dir / "sb_tensorboard/"),
        device=device,
    )

    logger.info("Stable-Baselines3 DQN model initialized.")
    logger.info(f"Model will be saved to: {save_path}")

    # --- 4. Training ---
    logger.info("Starting training...")
    # SB3 handles the entire training loop, including logging.
    model.learn(
        total_timesteps=CONFIG["TOTAL_TIMESTEPS"],
        log_interval=10,  # Log every 10 episodes
    )
    logger.success("Training finished.")

    # --- 5. Final Save ---
    model.save(save_path)
    logger.info(f"Final model saved to {save_path}")


if __name__ == "__main__":
    main()
