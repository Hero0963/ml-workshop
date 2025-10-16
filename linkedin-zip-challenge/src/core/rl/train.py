# src/core/rl/train.py
"""
Main training script for the DQN Agent with advanced saving and loading.

This script orchestrates the training process by:
1. Loading a pre-generated set of training puzzles.
2. Optionally loading a pre-trained model to resume training.
3. Initializing the environment and the DQN agent.
4. Running the main training loop, saving checkpoints and the best model.
5. Logging progress and saving the final trained model.

Requires:
- torch
- numpy
- loguru
- tqdm
- gymnasium
"""

import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# --- Add project root to sys.path for absolute imports ---
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- End of path setup ---

from src.core.rl.dqn_agent import DQNAgent  # noqa: E402
from src.core.rl.rl_env import PuzzleEnv  # noqa: E402
from src.core.utils import Puzzle  # noqa: E402

# --- Configuration ---
CONFIG: Dict[str, any] = {
    # --- Path Settings ---
    # IMPORTANT: Update this path to your generated .pkl file
    "DATASET_PATH": "D:/it_project/github_sync/ml-workshop/linkedin-zip-challenge/datasets/rl_datasets/rl_dataset_2025-10-12_092521/puzzles.pkl",
    "LOAD_MODEL_ON_START": False,  # Set to True to resume training
    "MODEL_LOAD_PATH": "models/dqn_agent_checkpoint.pth",
    "BEST_MODEL_SAVE_PATH": "models/dqn_agent_best.pth",
    "CHECKPOINT_SAVE_PATH": "models/dqn_agent_checkpoint.pth",
    # --- Training Settings ---
    "TOTAL_TIMESTEPS": 200_000,
    "LEARNING_RATE": 5e-4,
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.05,
    "EPSILON_DECAY": 50_000,
    # --- Agent & Buffer Settings ---
    "BATCH_SIZE": 128,
    "REPLAY_CAPACITY": 100_000,
    "TARGET_UPDATE_FREQ": 1000,
    "AGENT_TAU": 0.995,
    # --- Saving & Logging ---
    "SAVE_CHECKPOINT_FREQ": 10_000,  # Save a checkpoint every N steps
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
    log_dir = PROJECT_ROOT / "linkedin-zip-challenge" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "training_{time:YYYY-MM-DD_HH-mm-ss}.log"

    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(log_file_path, level="INFO")  # Add file sink

    logger.info("Starting DQN training process...")

    # Create full paths for models
    base_model_dir = PROJECT_ROOT / "linkedin-zip-challenge"
    load_path = base_model_dir / CONFIG["MODEL_LOAD_PATH"]
    best_save_path = base_model_dir / CONFIG["BEST_MODEL_SAVE_PATH"]
    checkpoint_save_path = base_model_dir / CONFIG["CHECKPOINT_SAVE_PATH"]
    best_save_path.parent.mkdir(parents=True, exist_ok=True)

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

    # --- 4. Initial Model Loading (Resume from Checkpoint) ---
    if CONFIG["LOAD_MODEL_ON_START"]:
        if load_path.exists():
            logger.info(f"Loading model from {load_path} to resume training...")
            state_dict = torch.load(load_path, map_location=device)
            agent.online_net.load_state_dict(state_dict)
            agent.target_net.load_state_dict(state_dict)  # Sync both networks
            logger.success("Model loaded successfully.")
        else:
            logger.warning(
                f"Load model was set to True, but no model found at {load_path}. Starting from scratch."
            )
    else:
        logger.info("Starting training from scratch.")

    # --- 5. Training Loop ---
    obs, _ = env.reset()
    processed_obs = agent.preprocess_obs(obs)
    episode_reward = 0
    episode_rewards: List[float] = []
    best_avg_reward = -float("inf")

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

        if len(agent.buffer) > CONFIG["BATCH_SIZE"]:
            agent.learn()

        # --- Checkpointing and Best Model Logic ---
        if done:
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > 10:  # Start checking only after a few episodes
                avg_reward = np.mean(episode_rewards[-100:])  # Avg of last 100

                if len(episode_rewards) % 10 == 0:
                    logger.info(
                        f"Step: {step}, Episodes: {len(episode_rewards)}, "
                        f"Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}"
                    )

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(agent.online_net.state_dict(), best_save_path)
                    logger.success(
                        f"New best model saved with avg reward: {best_avg_reward:.2f}"
                    )

            # Reset for next episode
            env = PuzzleEnv(random.choice(training_puzzles))
            obs, _ = env.reset()
            processed_obs = agent.preprocess_obs(obs)
            episode_reward = 0

        # Periodic checkpoint saving
        if step % CONFIG["SAVE_CHECKPOINT_FREQ"] == 0:
            torch.save(agent.online_net.state_dict(), checkpoint_save_path)
            logger.info(f"Periodic checkpoint saved at step {step}.")

    # --- 6. Final Save ---
    logger.success("Training finished.")
    torch.save(agent.online_net.state_dict(), checkpoint_save_path)
    logger.info(f"Final model checkpoint saved to {checkpoint_save_path}")


if __name__ == "__main__":
    main()
