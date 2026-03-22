"""
Tabular Q-Learning Agent implementation.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np

from board_game_rl.agents.base import BaseAgent
from board_game_rl.utils.logger import get_logger

logger = get_logger(__name__)


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning Agent.
    Maintains a Q-Table mapping state -> action -> value.
    """

    def __init__(
        self,
        name: str = "QLearningAgent",
        alpha: float = 0.1,  # 學習率
        gamma: float = 0.9,  # 折扣因子
        epsilon: float = 0.1,  # 探索機率
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-Table: Dict[state_str, Dict[action_str, float]]
        self.q_table: Dict[str, Dict[str, float]] = {}

    def _get_state_key(self, observation: np.ndarray) -> str:
        """Convert numpy array observation to a string key for the dictionary."""
        # 將 numpy array 攤平轉成 list 再轉字串，作為唯一的盤面 ID
        return str(observation.flatten().tolist())

    def _get_action_key(self, action: int) -> str:
        """Convert action integer to string key."""
        return str(action)

    def _get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value from table, default to 0.0 if not found."""
        if state_key not in self.q_table:
            return 0.0
        return self.q_table[state_key].get(action_key, 0.0)

    def act(
        self, observation: np.ndarray, info: dict, is_training: bool = False
    ) -> int:
        """
        Choose action using epsilon-greedy policy.
        """
        legal_actions = info.get("legal_actions", [])
        if not legal_actions:
            raise ValueError("No legal actions available.")

        # 探索 (Exploration): 隨機亂下
        if is_training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        # 利用 (Exploitation): 查表找最高分
        state_key = self._get_state_key(observation)
        best_actions = []
        max_q = float("-inf")

        for action in legal_actions:
            action_key = self._get_action_key(action)
            q_value = self._get_q_value(state_key, action_key)

            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)

        return random.choice(best_actions)

    def learn(
        self,
        old_observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_info: dict,
        done: bool,
    ) -> None:
        """
        Update Q-value using Bellman Equation.
        """
        state_key = self._get_state_key(old_observation)
        action_key = self._get_action_key(action)

        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        old_q = self._get_q_value(state_key, action_key)

        if done:
            # 遊戲結束，未來的分數是 0
            target_q = reward
        else:
            # 遊戲還沒結束，去看看「對手下完之後，輪到我下的最佳分數」是多少
            next_state_key = self._get_state_key(next_observation)
            next_legal_actions = next_info.get("legal_actions", [])

            max_next_q = float("-inf")
            for next_action in next_legal_actions:
                next_action_key = self._get_action_key(next_action)
                q_value = self._get_q_value(next_state_key, next_action_key)
                if q_value > max_next_q:
                    max_next_q = q_value

            if max_next_q == float("-inf"):
                max_next_q = 0.0

            # 貝爾曼方程式: 目標值 = 眼前的獎勵 + 打折後的未來最高獎勵
            target_q = reward + self.gamma * max_next_q

        # 更新 Q-Table
        self.q_table[state_key][action_key] = old_q + self.alpha * (target_q - old_q)

    def save_model(self, filepath: str) -> None:
        """Save Q-table to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.q_table, f)
        logger.info(f"Model saved to {filepath} (States learned: {len(self.q_table)})")

    def load_model(self, filepath: str) -> None:
        """Load Q-table from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.q_table = json.load(f)
            logger.info(
                f"Model loaded from {filepath} (States learned: {len(self.q_table)})"
            )
        else:
            logger.warning(f"Model {filepath} not found, starting fresh.")
