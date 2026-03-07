"""
Random Agent implementation for baseline testing.
"""

import random

import numpy as np

from board_game_rl.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that acts completely randomly from all available legal moves.
    """

    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name=name)

    def act(self, observation: np.ndarray, info: dict) -> int:
        """Pick a random action from legal actions."""
        legal_actions = info.get("legal_actions", [])
        if not legal_actions:
            raise ValueError("No legal actions available.")

        return random.choice(legal_actions)
