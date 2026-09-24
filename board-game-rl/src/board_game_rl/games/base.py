# src/board_game_rl/games/base.py
"""
Game rules interface consumed by search agents (MCTS now, AlphaZero later).

A search agent needs a forward model: "from this state, what can I play, and what
happens if I play it?". Games implement this Protocol; agents only depend on it,
so the same search code works for Tic-Tac-Toe today and Connect Four tomorrow.
"""

from collections.abc import Hashable
from typing import Protocol, TypeVar

import numpy as np

StateT = TypeVar("StateT", bound=Hashable)


class GameRules(Protocol[StateT]):
    """Pure-function rules of a two-player, zero-sum, perfect-information game.

    Players are 1 and -1. States must be immutable and hashable so a search tree
    can share them between nodes without defensive copies.
    """

    def initial_state(self) -> StateT: ...

    def to_move(self, state: StateT) -> int:
        """The player (1 or -1) whose turn it is."""
        ...

    def legal_actions(self, state: StateT) -> list[int]:
        """Playable actions. Only meaningful while `winner(state)` is None."""
        ...

    def next_state(self, state: StateT, action: int) -> StateT:
        """The state after `to_move(state)` plays `action`; `state` is untouched."""
        ...

    def winner(self, state: StateT) -> int | None:
        """1 or -1 if that player has won, 0 for a draw, None while still playing."""
        ...

    def from_observation(self, observation: np.ndarray, to_move: int) -> StateT:
        """Rebuild a state from an environment observation."""
        ...
