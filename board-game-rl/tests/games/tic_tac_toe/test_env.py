"""
Tests for Tic-Tac-Toe Gymnasium Environment.
"""

import numpy as np

from board_game_rl.games.tic_tac_toe.env import TicTacToeEnv


def test_env_initialization():
    env = TicTacToeEnv()
    obs, info = env.reset()

    assert env.action_space.n == 9
    assert obs.shape == (3, 3)
    assert np.all(obs == 0)
    assert "current_player" in info
    assert "legal_actions" in info
    assert len(info["legal_actions"]) == 9


def test_env_step_valid():
    env = TicTacToeEnv()
    env.reset()

    # Step at center (row 1, col 1 -> index 4)
    obs, reward, terminated, truncated, info = env.step(4)

    assert obs[1, 1] == 1
    assert reward == 0.0
    assert terminated is False
    assert info["current_player"] == -1
    assert 4 not in info["legal_actions"]


def test_env_step_invalid_penalty():
    env = TicTacToeEnv()
    env.reset()

    env.step(4)  # P1 plays 4
    obs, reward, terminated, truncated, info = env.step(4)  # P2 tries to play 4 again

    # Invalid move ends game immediately with a penalty
    assert reward == -10.0
    assert terminated is True
