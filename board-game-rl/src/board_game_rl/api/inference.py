"""
Inference logic for Tic-Tac-Toe.
"""

import numpy as np

from board_game_rl.agents.alphabeta_agent import AlphaBetaAgent
from board_game_rl.core.engine import TicTacToeEngine


def get_optimal_move(board_state: list[list[int]], current_player: int) -> int:
    """
    Given a board state and the current player, return the optimal action (0-8).
    Currently implemented with AlphaBetaAgent.
    """
    engine = TicTacToeEngine()
    engine.board = board_state
    engine.current_player = current_player
    legal_actions = engine.get_legal_actions()

    # If no legal actions, return an invalid move indicator or handle it
    if not legal_actions:
        return -1

    info = {"legal_actions": [r * 3 + c for r, c in legal_actions]}

    # Use AlphaBetaAgent to find the best move
    agent = AlphaBetaAgent(player=current_player)
    action = agent.act(np.array(board_state), info)

    return action
