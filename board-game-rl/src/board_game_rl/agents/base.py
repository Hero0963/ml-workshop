"""
Base Agent interface.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all board game agents."""

    def __init__(self, name: str = "Agent"):
        self.name = name

    @abstractmethod
    def act(self, observation: np.ndarray, info: dict) -> int:
        """
        Choose an action based on the current observation.

        Args:
            observation: State of the environment.
            info: Additional information from the environment.
                  (e.g., 'legal_actions': [0, 1, 2...])

        Returns:
            The chosen action.
        """
        pass
