# src/core/rl/train_single_cnn_sb.py
"""
Training script for a single puzzle using the Stable-Baselines3 library
and a CNN policy.

This script is for overfitting on one puzzle to test the CNN-compatible
environment and reward function.

Key Features:
1. Uses `stable_baselines3.DQN` with a `"CnnPolicy"`.
2. The custom environment `PuzzleEnv` is now expected to return image-like
   observations.
3. `FlattenObservation` wrapper is removed as it's no longer needed.
4. Model save paths are unique to this script.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import gymnasium as gym
from loguru import logger
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# --- Add project root to sys.path for absolute imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---

from src.core.rl.rl_env import PuzzleEnv  # noqa: E402
from src.core.utils import Puzzle  # noqa: E402


class CustomCnn(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the 6x6 puzzle environment.
    The default NatureCNN is too large for our small input.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# --- Configuration ---
CONFIG: Dict[str, any] = {
    # --- Path Settings ---
    "DATASET_PATH": "D:/it_project/github_sync/ml-workshop/linkedin-zip-challenge/datasets/rl_datasets/rl_dataset_2025-10-12_092521/puzzles.pkl",
    "MODEL_SAVE_PATH": "models/dqn_sb_cnn_single.zip",  # New model name for CNN
    # --- Training Settings ---
    "TOTAL_TIMESTEPS": 300_000,  # Increased for CNN
    "LEARNING_RATE": 1e-4,  # Often lower for CNNs
    "GAMMA": 0.99,
    "EPSILON_DECAY_STEPS": 100_000,  # Slower decay
    # --- Agent & Buffer Settings ---
    "BATCH_SIZE": 128,
    "REPLAY_CAPACITY": 100_000,
    "TARGET_UPDATE_FREQ": 1000,  # In steps
    "AGENT_TAU": 0.995,
    "DISTANCE_REWARD_WEIGHT": 0.01,  # Weight for the distance-based reward shaping
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
    log_file_path = log_dir / "training_sb_cnn_single_{time:YYYY-MM-DD_HH-mm-ss}.log"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file_path, level="INFO")

    logger.info("Starting Stable-Baselines3 CNN single-puzzle overfitting test...")

    base_model_dir = PROJECT_ROOT / "linkedin-zip-challenge"
    save_path = base_model_dir / CONFIG["MODEL_SAVE_PATH"]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- 2. Puzzle and Environment Initialization ---
    dataset_path = Path(CONFIG["DATASET_PATH"])
    training_puzzles = load_puzzles(dataset_path)
    fixed_puzzle = training_puzzles[0]

    # The new PuzzleEnv returns image-like observations, so FlattenObservation is not needed.
    def make_env():
        env = PuzzleEnv(
            fixed_puzzle, distance_reward_weight=CONFIG["DISTANCE_REWARD_WEIGHT"]
        )
        return env

    env = make_vec_env(make_env, n_envs=1)
    logger.info("Created CNN-compatible environment for Stable-Baselines3.")

    # --- 3. Agent Initialization ---
    exploration_fraction = CONFIG["EPSILON_DECAY_STEPS"] / CONFIG["TOTAL_TIMESTEPS"]

    policy_kwargs = {
        "features_extractor_class": CustomCnn,
        "normalize_images": False,
    }

    # Use "CnnPolicy" and tell it not to normalize our already-normalized input
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=CONFIG["LEARNING_RATE"],
        buffer_size=CONFIG["REPLAY_CAPACITY"],
        learning_starts=CONFIG["BATCH_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        tau=CONFIG["AGENT_TAU"],
        gamma=CONFIG["GAMMA"],
        train_freq=(1, "step"),
        target_update_interval=CONFIG["TARGET_UPDATE_FREQ"],
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=str(log_dir / "sb_tensorboard/"),
        policy_kwargs=policy_kwargs,  # Use the custom CNN
        device=device,
    )

    logger.info("Stable-Baselines3 DQN model with CustomCnn policy initialized.")
    logger.info(f"Model will be saved to: {save_path}")

    # --- 4. Training ---
    logger.info("Starting training...")
    model.learn(
        total_timesteps=CONFIG["TOTAL_TIMESTEPS"],
        log_interval=10,
    )
    logger.success("Training finished.")

    # --- 5. Final Save ---
    model.save(save_path)
    logger.info(f"Final model saved to {save_path}")


if __name__ == "__main__":
    main()
