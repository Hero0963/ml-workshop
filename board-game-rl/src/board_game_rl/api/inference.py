# src/board_game_rl/api/inference.py
"""
Inference logic for Tic-Tac-Toe.
"""

from pathlib import Path

import numpy as np

from board_game_rl.agents.q_learning_agent import QLearningAgent
from board_game_rl.agents.random_agent import RandomAgent
from board_game_rl.games.tic_tac_toe.alphabeta_agent import AlphaBetaAgent
from board_game_rl.games.tic_tac_toe.engine import TicTacToeEngine
from board_game_rl.utils.logger import get_logger

logger = get_logger(__name__)

_agent_cache: dict[str, QLearningAgent | AlphaBetaAgent | RandomAgent] = {}


def _resolve_q_table_path() -> Path:
    """Resolve Q-table model path (Docker or local)."""
    docker_path = Path("/app/models/q_table.json")
    if docker_path.exists():
        return docker_path
    return Path(__file__).parent.parent.parent.parent / "models" / "q_table.json"


def _get_agent(
    agent_type: str, player: int
) -> QLearningAgent | AlphaBetaAgent | RandomAgent:
    """Return a cached agent instance, creating and loading on first access."""
    cache_key = f"{agent_type}:{player}"

    if cache_key in _agent_cache:
        return _agent_cache[cache_key]

    if "Q-Learning" in agent_type:
        agent = QLearningAgent(player=player)
        model_path = _resolve_q_table_path()
        if model_path.exists():
            agent.load_model(str(model_path))
        else:
            logger.warning(
                f"Q-Table not found at {model_path}. Agent will play randomly."
            )
    elif "Random" in agent_type:
        agent = RandomAgent()
    else:
        agent = AlphaBetaAgent(player=player)

    _agent_cache[cache_key] = agent
    return agent


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

    if not legal_actions:
        return -1

    info = {"legal_actions": [r * 3 + c for r, c in legal_actions]}
    agent = _get_agent(agent_type, current_player)

    if isinstance(agent, QLearningAgent):
        return agent.act(np.array(board_state), info, is_training=False)
    return agent.act(np.array(board_state), info)
