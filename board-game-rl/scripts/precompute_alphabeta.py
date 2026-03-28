"""
Pre-computes the optimal moves for all reachable Tic-Tac-Toe states using Alpha-Beta Pruning.
This allows for instantaneous O(1) lookups during inference.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add src to python path for relative imports to work
sys.path.append(str(Path(__file__).parent.parent / "src"))

from board_game_rl.games.tic_tac_toe.engine import TicTacToeEngine
from board_game_rl.games.tic_tac_toe.alphabeta_agent import AlphaBetaAgent


def get_state_key(board: list[list[int]]) -> str:
    """Convert 2D board to string key."""
    return str(np.array(board).flatten().tolist())


def precompute_states():
    """
    Explore all possible game states using Depth-First Search (DFS)
    and compute the optimal move for each state using Alpha-Beta.
    """
    engine = TicTacToeEngine()
    cache: Dict[str, int] = {}
    visited = set()

    # We need an agent for X (1) and O (-1)
    agent_x = AlphaBetaAgent(player=1)
    agent_o = AlphaBetaAgent(player=-1)

    print("Starting DFS to pre-compute all Alpha-Beta moves...")

    def dfs(current_engine: TicTacToeEngine):
        state_key = get_state_key(current_engine.board)

        # If game is over or state already evaluated, return
        if current_engine.winner is not None or state_key in visited:
            return

        visited.add(state_key)

        # Compute best move for the current player
        legal_actions = current_engine.get_legal_actions()
        info = {"legal_actions": [r * 3 + c for r, c in legal_actions]}

        if current_engine.current_player == 1:
            best_action = agent_x.act(np.array(current_engine.board), info)
        else:
            best_action = agent_o.act(np.array(current_engine.board), info)

        cache[state_key] = int(best_action)

        # Recurse into all possible next states
        for action in legal_actions:
            # Create a copy of the engine to simulate the move
            next_engine = TicTacToeEngine()
            next_engine.board = [row[:] for row in current_engine.board]
            next_engine.current_player = current_engine.current_player

            # Make the move and continue DFS
            next_engine.step(action)
            dfs(next_engine)

    # Start DFS from the empty board
    dfs(engine)

    print(f"Pre-computation complete! Explored and cached {len(cache)} unique states.")

    # Save the cache to disk
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    cache_path = models_dir / "alphabeta_cache.json"

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    print(f"Saved Alpha-Beta cache to: {cache_path}")


if __name__ == "__main__":
    # Ensure tqdm is available
    precompute_states()
