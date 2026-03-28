"""
Gymnasium environment wrapper for Tic-Tac-Toe.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from board_game_rl.games.tic_tac_toe.engine import TicTacToeEngine


class TicTacToeEnv(gym.Env):
    """
    TicTacToe Environment that follows gym interface.
    Observation: 3x3 grid (1 for X, -1 for O, 0 for empty)
    Action: Discrete scalar from 0 to 8 representing cell index (row * 3 + col)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.engine = TicTacToeEngine()
        self.render_mode = render_mode

        # Action space: 0-8 (9 cells)
        self.action_space = spaces.Discrete(9)

        # Observation space: 3x3 matrix, values in [-1, 1]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step within the environment.
        """
        row, col = action // 3, action % 3

        # Keep track of current player before move (for reward assignment)
        player_making_move = self.engine.current_player

        valid = self.engine.step((row, col))

        terminated = False
        reward = 0.0

        if not valid:
            # Invalid move ends the game with a penalty
            terminated = True
            reward = -10.0
        elif self.engine.winner is not None:
            terminated = True
            if self.engine.winner == player_making_move:
                reward = 1.0  # Win
            elif self.engine.winner == 0:
                reward = 0.0  # Draw
            else:
                reward = (
                    -1.0
                )  # Should theoretically not happen on current player's turn in TTT

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self.engine.render()
        elif self.render_mode == "human":
            print(self.engine.render())
            print(f"Current Player: {'X' if self.engine.current_player == 1 else 'O'}")
            print()
            return None
        return None

    def _get_obs(self) -> np.ndarray:
        return np.array(self.engine.board, dtype=np.int8)

    def _get_info(self) -> dict[str, Any]:
        return {
            "current_player": self.engine.current_player,
            "legal_actions": [r * 3 + c for r, c in self.engine.get_legal_actions()],
        }
