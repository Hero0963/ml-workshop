# src\core\rl\dqn_agent.py
"""
This module contains the core components for a DQN Agent.
Inspired by the structures in the user's chap_02_dqn notebooks.

Requires:
- torch
- numpy
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn, optim


class DQNModel(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) to approximate the Q-function.
    It takes a flattened observation and outputs Q-values for each action.
    """

    def __init__(self, obs_shape: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """A simple replay buffer using a deque."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        # Note: The `*zip(*...)` trick is a concise way to transpose the list of tuples
        # from (obs, action, ...), (obs, action, ...), ...
        # to (obs, obs, ...), (action, action, ...), ...
        obses, actions, rewards, next_obses, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(obses),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obses),
            np.array(dones, dtype=np.uint8),
        )


class DQNAgent:
    """
    The main agent class that orchestrates the model, replay buffer, and learning process.
    """

    def __init__(
        self,
        obs_shape: int,
        n_actions: int,
        device: torch.device,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        replay_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        tau: float = 1.0,  # 1.0 for hard update, < 1.0 for soft update
    ):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        self.online_net = DQNModel(obs_shape, n_actions).to(device)
        self.target_net = DQNModel(obs_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network is only for evaluation

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(replay_capacity)

        self._learn_step_counter = 0

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(obs_tensor)
            return q_values.argmax().item()

    def learn(self) -> None:
        """
        Performs one step of learning. Samples a batch from the replay buffer,
        calculates the loss, and updates the online network.
        """
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples to learn yet

        self._learn_step_counter += 1

        # --- 1. Sample a batch from the replay buffer ---
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_obs_tensor = torch.from_numpy(next_obs).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).float().to(self.device)

        # --- 2. Calculate the Q-values for the current state ---
        # We need to get the Q-values for the actions that were actually taken
        q_values = self.online_net(obs_tensor)
        state_action_values = q_values.gather(1, actions_tensor.unsqueeze(-1)).squeeze(
            -1
        )

        # --- 3. Calculate the target Q-values for the next state ---
        with torch.no_grad():
            # Using the target network for stability
            next_state_q_values = self.target_net(next_obs_tensor).max(1)[0]
            # If the episode was done, the future reward is 0
            next_state_q_values[dones_tensor == 1] = 0.0
            # Bellman equation
            expected_state_action_values = (
                rewards_tensor + self.gamma * next_state_q_values
            )

        # --- 4. Compute the loss and update the online network ---
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # --- 5. Update the target network ---
        if self._learn_step_counter % self.target_update_freq == 0:
            self._update_target_net()
            logger.debug(f"Step {self._learn_step_counter}: Target network updated.")

    def _update_target_net(self):
        """Update the target network weights with the online network weights."""
        if self.tau == 1.0:  # Hard update
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:  # Soft update
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                )

    @staticmethod
    def preprocess_obs(obs: dict) -> np.ndarray:
        """
        Flattens and concatenates the observation dictionary into a single numpy array.
        This is a simple preprocessing step. More complex ones could be used.
        """
        return np.concatenate(
            [obs["agent_location"].flatten(), obs["next_waypoint_location"].flatten()]
        )
