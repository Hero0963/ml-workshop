"""
Tic-Tac-Toe Game Engine.
Independent of outside RL frameworks to maintain pure game logic.
"""

from typing import Literal

# Types
Player = Literal[1, -1]  # 1 for X, -1 for O
Board = list[list[int]]  # 3x3 grid, 0 for empty


class TicTacToeEngine:
    """Core logic for Tic-Tac-Toe."""

    def __init__(self) -> None:
        self.board: Board = [[0, 0, 0] for _ in range(3)]
        self.current_player: Player = 1
        self.winner: Player | None | Literal[0] = None  # 0 for draw

    def reset(self) -> None:
        """Reset the game state."""
        self.board = [[0, 0, 0] for _ in range(3)]
        self.current_player = 1
        self.winner = None

    def get_legal_actions(self) -> list[tuple[int, int]]:
        """Return list of empty cells (row, col)."""
        legal_actions = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 0:
                    legal_actions.append((r, c))
        return legal_actions

    def step(self, action: tuple[int, int]) -> bool:
        """
        Apply an action.
        Returns:
            bool: True if valid move, False otherwise.
        """
        if self.winner is not None:
            return False  # Game over

        r, c = action
        if self.board[r][c] != 0:
            return False  # Invalid move

        self.board[r][c] = self.current_player
        self._check_winner()

        if self.winner is None:
            # Switch player
            self.current_player = -1 if self.current_player == 1 else 1

        return True

    def _check_winner(self) -> None:
        """Check all win conditions and draws."""
        # Rows and cols
        for i in range(3):
            if abs(sum(self.board[i])) == 3:
                self.winner = self.board[i][0]
                return
            if abs(self.board[0][i] + self.board[1][i] + self.board[2][i]) == 3:
                self.winner = self.board[0][i]
                return

        # Diagonals
        if abs(self.board[0][0] + self.board[1][1] + self.board[2][2]) == 3:
            self.winner = self.board[0][0]
            return
        if abs(self.board[0][2] + self.board[1][1] + self.board[2][0]) == 3:
            self.winner = self.board[0][2]
            return

        # Draw check
        if not self.get_legal_actions():
            self.winner = 0

    def render(self) -> str:
        """String representation of the board."""
        chars = {1: "X", -1: "O", 0: " "}
        lines = []
        for row in self.board:
            lines.append(" | ".join(chars[cell] for cell in row))
            lines.append("-" * 9)
        return "\n".join(lines[:-1])
