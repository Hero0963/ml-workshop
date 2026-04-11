# src/board_game_rl/agents/dqn_agent.py
"""
Deep Q-Network (DQN) Agent implementation.

Replaces the Q-table with a neural network for Q-value approximation.
Key components:
  - DQNNetwork: simple MLP (9 → 128 → 128 → 9)
  - ReplayBuffer: experience replay to break data correlation
  - DQNAgent: extends BaseAgent with learn(), target network sync, save/load
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from board_game_rl.agents.base import BaseAgent
from board_game_rl.utils.logger import get_logger

logger = get_logger(__name__)


class DQNNetwork(nn.Module):
    """Simple MLP for Q-value approximation on a 3x3 board."""

    def __init__(
        self, input_size: int = 9, hidden_size: int = 128, output_size: int = 9
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer storing (state, action, reward, next_state, done, legal_actions)."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: list[int],
    ) -> None:
        self.buffer.append(
            (state, action, reward, next_state, done, next_legal_actions)
        )

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent.

    Uses a neural network instead of a Q-table for value approximation.
    Supports experience replay and target network for stable training.
    """

    def __init__(
        self,
        name: str = "DQNAgent",
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        player: int = 1,
        batch_size: int = 64,
        target_update_freq: int = 500,
        buffer_capacity: int = 50_000,
        device: str | None = None,
    ):
        super().__init__(name=name)
        self.gamma = gamma
        self.epsilon = epsilon
        self.player = player
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Policy network (the one we train)
        self.policy_net = DQNNetwork().to(self.device)
        # Target network (frozen copy, updated periodically)
        self.target_net = DQNNetwork().to(self.device)
        self.sync_target()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.learn_step_count = 0

    def sync_target(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _normalize_obs(self, observation: np.ndarray) -> np.ndarray:
        """Normalize board so agent always sees itself as player 1."""
        if self.player == 1:
            return observation
        return observation * -1

    def _obs_to_tensor(self, observation: np.ndarray) -> torch.Tensor:
        """Convert observation to a float tensor on device."""
        normalized = self._normalize_obs(observation).flatten().astype(np.float32)
        return torch.tensor(normalized, device=self.device).unsqueeze(0)

    def act(
        self, observation: np.ndarray, info: dict, is_training: bool = False
    ) -> int:
        """Choose action using epsilon-greedy over network Q-values."""
        legal_actions = info.get("legal_actions", [])
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if is_training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            q_values = self.policy_net(self._obs_to_tensor(observation)).squeeze(0)

        # Mask illegal actions with -inf
        mask = torch.full((9,), float("-inf"), device=self.device)
        for a in legal_actions:
            mask[a] = 0.0
        masked_q = q_values + mask

        return int(masked_q.argmax().item())

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_legal_actions: list[int],
    ) -> float | None:
        """
        Store transition and train on a mini-batch if buffer is large enough.

        Returns the batch loss, or None if buffer too small.
        """
        normalized_state = self._normalize_obs(state).flatten().astype(np.float32)
        normalized_next = self._normalize_obs(next_state).flatten().astype(np.float32)

        self.replay_buffer.push(
            normalized_state, action, reward, normalized_next, done, next_legal_actions
        )

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_legals = zip(*batch)

        states_t = torch.tensor(np.array(states), device=self.device)
        actions_t = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values for chosen actions
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q_all = self.target_net(next_states_t)

            # Mask illegal actions for each sample in batch
            next_q_max = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.batch_size):
                legal = next_legals[i]
                if legal and not dones[i]:
                    legal_q = next_q_all[i][legal]
                    next_q_max[i] = legal_q.max()

            target_q = rewards_t + self.gamma * next_q_max * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.learn_step_count += 1
        if self.learn_step_count % self.target_update_freq == 0:
            self.sync_target()

        return float(loss.item())

    def save_model(self, filepath: str) -> None:
        """Save policy network weights."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        logger.info(f"DQN model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load policy network weights and sync target."""
        self.policy_net.load_state_dict(
            torch.load(filepath, map_location=self.device, weights_only=True)
        )
        self.sync_target()
        logger.info(f"DQN model loaded from {filepath}")
