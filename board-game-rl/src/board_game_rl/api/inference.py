"""
Inference logic for Tic-Tac-Toe.
"""

import os
from pathlib import Path

import numpy as np

from board_game_rl.agents.random_agent import RandomAgent
from board_game_rl.agents.q_learning_agent import QLearningAgent
from board_game_rl.games.tic_tac_toe.alphabeta_agent import AlphaBetaAgent
from board_game_rl.games.tic_tac_toe.engine import TicTacToeEngine


def get_optimal_move(
    board_state: list[list[int]],
    current_player: int,
    agent_type: str = "Alpha-Beta Pruning (完美大師)",
) -> int:
    """
    Given a board state and the current player, return the optimal action (0-8).
    """
    engine = TicTacToeEngine()
    engine.board = board_state
    engine.current_player = current_player
    legal_actions = engine.get_legal_actions()

    # If no legal actions, return an invalid move indicator or handle it
    if not legal_actions:
        return -1

    info = {"legal_actions": [r * 3 + c for r, c in legal_actions]}

    if "Q-Learning" in agent_type:
        agent = QLearningAgent()

        # Try to load the trained model
        # Default path relative to this file
        model_path = (
            Path(__file__).parent.parent.parent.parent / "models" / "q_table.json"
        )

        # If running in docker, the path is /app/models/q_table.json
        if os.path.exists("/app/models/q_table.json"):
            model_path = Path("/app/models/q_table.json")

        if model_path.exists():
            agent.load_model(str(model_path))
        else:
            print(
                f"Warning: Q-Table not found at {model_path}. Agent will play randomly."
            )

        action = agent.act(np.array(board_state), info, is_training=False)
    elif "Random" in agent_type:
        agent = RandomAgent()
        action = agent.act(np.array(board_state), info)
    else:
        # Default to AlphaBetaAgent
        agent = AlphaBetaAgent(player=current_player)
        action = agent.act(np.array(board_state), info)

    return action
