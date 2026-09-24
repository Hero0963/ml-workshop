# src/board_game_rl/games/tic_tac_toe/rules.py
"""
Immutable Tic-Tac-Toe rules for search agents (implements `GameRules`).

`TicTacToeEngine` stays the source of truth for the UI and the Gym env. This module
mirrors its rules on tuples, which are cheap to branch: a search tree creates a new
state per move instead of copying and mutating an engine.
"""

from typing import NamedTuple

import numpy as np

BOARD_CELLS = 9
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
DRAW = 0

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class TicTacToeState(NamedTuple):
    board: tuple[int, ...]  # 9 cells, row-major; 1 = X, -1 = O, 0 = empty
    to_move: int


class TicTacToeRules:
    """Pure-function Tic-Tac-Toe rules; see `GameRules` for the contract."""

    def initial_state(self) -> TicTacToeState:
        return TicTacToeState(board=(EMPTY,) * BOARD_CELLS, to_move=PLAYER_X)

    def to_move(self, state: TicTacToeState) -> int:
        return state.to_move

    def legal_actions(self, state: TicTacToeState) -> list[int]:
        return [cell for cell, value in enumerate(state.board) if value == EMPTY]

    def next_state(self, state: TicTacToeState, action: int) -> TicTacToeState:
        if state.board[action] != EMPTY:
            raise ValueError(f"Cell {action} is already occupied.")
        board = state.board[:action] + (state.to_move,) + state.board[action + 1 :]
        return TicTacToeState(board=board, to_move=-state.to_move)

    def winner(self, state: TicTacToeState) -> int | None:
        board = state.board
        for a, b, c in WIN_LINES:
            if board[a] != EMPTY and board[a] == board[b] == board[c]:
                return board[a]
        if EMPTY not in board:
            return DRAW
        return None

    def from_observation(self, observation: np.ndarray, to_move: int) -> TicTacToeState:
        board = tuple(int(value) for value in np.asarray(observation).flatten())
        return TicTacToeState(board=board, to_move=to_move)
