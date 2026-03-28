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


# ── D4 Symmetry: 8 transforms for 3x3 board (4 rotations × 2 reflections) ──


def _precompute_symmetry_maps() -> list[tuple[np.ndarray, list[int]]]:
    """Precompute action/inverse maps for all 8 D4 symmetries.

    Returns list of (action_map, inverse_map):
      - action_map:  transformed_obs = original_flat[action_map]
      - inverse_map: inverse_map[original_action] = transformed_action
    First entry is always the identity transform.
    """
    idx = np.arange(9).reshape(3, 3)
    results: list[tuple[np.ndarray, list[int]]] = []
    for k in range(4):
        for flip in [False, True]:
            t = np.rot90(idx, k=k)
            if flip:
                t = np.fliplr(t)
            action_map = t.flatten()
            inverse = [0] * 9
            for t_idx, o_idx in enumerate(action_map):
                inverse[o_idx] = t_idx
            results.append((action_map, inverse))
    return results


_SYMMETRY_MAPS = _precompute_symmetry_maps()


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
        player: int = 1,  # 1 for X, -1 for O
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player = player

        # Q-Table: Dict[state_str, Dict[action_str, float]]
        self.q_table: Dict[str, Dict[str, float]] = {}

    def _normalize_obs(self, observation: np.ndarray) -> np.ndarray:
        """Normalize board so agent always sees itself as player 1."""
        if self.player == 1:
            return observation
        return observation * -1

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
        Uses D4 symmetry lookup: if the exact board isn't in the Q-table,
        check all 8 rotations/reflections for a match.
        """
        legal_actions = info.get("legal_actions", [])
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if is_training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        original_flat = self._normalize_obs(observation).flatten()

        # Try all 8 D4 symmetries (identity first, then rotations/reflections)
        for action_map, inverse_map in _SYMMETRY_MAPS:
            state_key = str(original_flat[action_map].tolist())

            if state_key not in self.q_table:
                continue

            # Found a match — pick best action via this transform's Q-values
            best_actions = []
            max_q = float("-inf")
            for action in legal_actions:
                q_value = self.q_table[state_key].get(
                    str(inverse_map[action]),
                    0.0,
                )
                if q_value > max_q:
                    max_q = q_value
                    best_actions = [action]
                elif q_value == max_q:
                    best_actions.append(action)
            return random.choice(best_actions)

        # No symmetric match found — fall back to random
        return random.choice(legal_actions)

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
        state_key = self._get_state_key(self._normalize_obs(old_observation))
        action_key = self._get_action_key(action)

        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        old_q = self._get_q_value(state_key, action_key)

        if done:
            # 遊戲結束，未來的分數是 0
            target_q = reward
        else:
            # 遊戲還沒結束，去看看「對手下完之後，輪到我下的最佳分數」是多少
            next_state_key = self._get_state_key(self._normalize_obs(next_observation))
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
