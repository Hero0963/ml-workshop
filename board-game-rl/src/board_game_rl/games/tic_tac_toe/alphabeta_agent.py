"""
Alpha-Beta Pruning Agent for Tic-Tac-Toe.
"""

from typing import Literal

import numpy as np

from board_game_rl.agents.base import BaseAgent
from board_game_rl.utils.logger import get_logger

logger = get_logger(__name__)

Player = Literal[1, -1]
Board = list[list[int]]


class AlphaBetaAgent(BaseAgent):
    """
    Minimax agent with Alpha-Beta Pruning.
    """

    def __init__(self, name: str = "AlphaBeta", player: Player = -1) -> None:
        """
        Initialize the agent.

        Args:
            name: The name of the agent.
            player: The player this agent represents (1 for X, -1 for O). Defaults to -1.
        """
        super().__init__(name=name)
        self.player = player

    def _check_winner(self, board: Board) -> Player | Literal[0] | None:
        """Check the winner of the current board state."""
        # Rows and cols
        for i in range(3):
            if abs(sum(board[i])) == 3:
                return 1 if sum(board[i]) > 0 else -1
            col_sum = board[0][i] + board[1][i] + board[2][i]
            if abs(col_sum) == 3:
                return 1 if col_sum > 0 else -1

        # Diagonals
        diag1 = board[0][0] + board[1][1] + board[2][2]
        if abs(diag1) == 3:
            return 1 if diag1 > 0 else -1
        diag2 = board[0][2] + board[1][1] + board[2][0]
        if abs(diag2) == 3:
            return 1 if diag2 > 0 else -1

        # Draw check
        has_empty = any(0 in row for row in board)
        if not has_empty:
            return 0  # Draw

        return None  # Ongoing

    def _evaluate(self, board: Board) -> float:
        """Static evaluation of the board state."""
        winner = self._check_winner(board)
        if winner == self.player:
            return 10.0
        elif winner == -self.player:
            return -10.0
        elif winner == 0:
            return 0.0
        return 0.0

    def _get_legal_actions(self, board: Board) -> list[tuple[int, int]]:
        """Return list of empty cells as coordinates."""
        return [(r, c) for r in range(3) for c in range(3) if board[r][c] == 0]

    def alphabeta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
    ) -> tuple[float, tuple[int, int] | None]:
        """
        Recursive Alpha-Beta search.

        Returns:
            A tuple of (best_score, best_action_coordinates)
        """
        winner = self._check_winner(board)
        indent = "  " * depth
        role = "🤖 AI (MAX)" if is_maximizing else "👤 Player (MIN)"

        if winner is not None:
            score = self._evaluate(board)
            return score, None

        legal_actions = self._get_legal_actions(board)
        if not legal_actions:
            return 0.0, None

        best_action = None

        if is_maximizing:
            best_score = -float("inf")
            for i, action in enumerate(legal_actions):
                r, c = action
                board[r][c] = self.player

                if depth < 2:  # Print first two levels of thought process
                    logger.info(f"{indent}┣━ {role} 嘗試下在 {action} ...")

                score, _ = self.alphabeta(board, depth + 1, alpha, beta, False)

                # Penalize delayed wins to encourage faster finishes
                if score > 0:
                    score -= 0.1
                elif score < 0:
                    score += 0.1

                board[r][c] = 0

                if score > best_score:
                    best_score = score
                    best_action = action

                alpha = max(alpha, best_score)
                if depth < 2:
                    logger.info(
                        f"{indent}┃  ┗━ 預期最終得分: {score:.1f} ➔ (目前保底 α={alpha:.1f}, 防線 β={beta:.1f})"
                    )

                if alpha >= beta:
                    if depth < 2:
                        logger.info(
                            f"{indent}✂️ ┗━ 【剪枝發生】對手已有更好對策 (α≥β)，跳過剩餘 {len(legal_actions) - i - 1} 種可能性！"
                        )
                    break
            return best_score, best_action

        else:
            best_score = float("inf")
            opponent = -self.player
            for i, action in enumerate(legal_actions):
                r, c = action
                board[r][c] = opponent

                if depth < 2:
                    logger.info(f"{indent}┣━ {role} 嘗試下在 {action} ...")

                score, _ = self.alphabeta(board, depth + 1, alpha, beta, True)

                # Penalize delayed losses
                if score > 0:
                    score -= 0.1
                elif score < 0:
                    score += 0.1

                board[r][c] = 0

                if score < best_score:
                    best_score = score
                    best_action = action

                beta = min(beta, best_score)
                if depth < 2:
                    logger.info(
                        f"{indent}┃  ┗━ 預期最終得分: {score:.1f} ➔ (目前保底 α={alpha:.1f}, 防線 β={beta:.1f})"
                    )

                if alpha >= beta:
                    if depth < 2:
                        logger.info(
                            f"{indent}✂️ ┗━ 【剪枝發生】對策太差 (α≥β)，跳過剩餘 {len(legal_actions) - i - 1} 種可能性！"
                        )
                    break
            return best_score, best_action

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        """
        Choose an action based on the Minimax evaluation.

        Args:
            observation: State of the environment (not directly used here, we parse 'info' state or cast).
            info: Current context. Here we expect 'board' representation or we cast observation.

        Returns:
            The chosen action index (0-8)
        """
        # Convert observation (which is 1D or 2D) to our 2D Board format
        obs_1d = observation.flatten()
        board_state: Board = [
            [int(obs_1d[0]), int(obs_1d[1]), int(obs_1d[2])],
            [int(obs_1d[3]), int(obs_1d[4]), int(obs_1d[5])],
            [int(obs_1d[6]), int(obs_1d[7]), int(obs_1d[8])],
        ]

        player_str = "X" if self.player == 1 else "O"
        logger.info(
            "\n" + "=" * 40 + f"\n🧠 第 {player_str} 方 (AI) 開始腦內推演 (Alpha-Beta)"
        )
        best_score, best_action = self.alphabeta(
            board=board_state,
            depth=0,
            alpha=-float("inf"),
            beta=float("inf"),
            is_maximizing=True,
        )

        if best_action is None:
            # Fallback if somehow no action
            legal = self._get_legal_actions(board_state)
            if legal:
                best_action = legal[0]
            else:
                return 0

        r, c = best_action
        action_idx = r * 3 + c
        logger.info(
            f"✨ 推演結束！決定走 {best_action}，預期得分：{best_score:.1f}\n"
            + "=" * 40
            + "\n"
        )
        return int(action_idx)
