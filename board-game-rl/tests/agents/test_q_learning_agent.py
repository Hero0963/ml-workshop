# tests/agents/test_q_learning_agent.py

import random
from pathlib import Path

import numpy as np
import pytest

from board_game_rl.agents.base import BaseAgent
from board_game_rl.agents.q_learning_agent import QLearningAgent
from board_game_rl.games.tic_tac_toe.alphabeta_agent import AlphaBetaAgent
from board_game_rl.games.tic_tac_toe.env import TicTacToeEnv

TRAINED_Q_TABLE = Path(__file__).resolve().parents[2] / "models" / "q_table.json"


def _key(board: list[int]) -> str:
    return str(board)


def _play_one_game(agent_x: BaseAgent, agent_o: BaseAgent) -> int:
    """Play a full game and return the winner (1 = X, -1 = O, 0 = draw)."""
    env = TicTacToeEnv()
    obs, info = env.reset()
    agents = {1: agent_x, -1: agent_o}
    terminated = False
    while not terminated:
        action = agents[info["current_player"]].act(obs, info)
        obs, _, terminated, _, info = env.step(action)
    return env.engine.winner


# ── Board normalization ──────────────────────────────────────────────────────


def test_normalize_obs_keeps_board_for_x():
    """Player X already sees itself as 1, so the board is unchanged."""
    agent = QLearningAgent(player=1)
    board = np.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(agent._normalize_obs(board), board)


def test_normalize_obs_flips_board_for_o():
    """Player O sees its own stones as 1, so one table serves both sides."""
    agent = QLearningAgent(player=-1)
    board = np.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(agent._normalize_obs(board), -board)


# ── Acting ───────────────────────────────────────────────────────────────────


def test_act_raises_without_legal_actions():
    """A full board has no move to choose."""
    agent = QLearningAgent()
    with pytest.raises(ValueError):
        agent.act(np.zeros((3, 3), dtype=np.int8), {"legal_actions": []})


def test_act_falls_back_to_legal_random_move_on_unknown_state():
    """An empty table means every state is unseen, but the move must stay legal."""
    random.seed(0)
    agent = QLearningAgent()
    legal = [2, 5, 7]
    for _ in range(20):
        assert (
            agent.act(np.zeros((3, 3), dtype=np.int8), {"legal_actions": legal})
            in legal
        )


def test_act_picks_highest_q_value():
    """With an exact table hit, the greedy move is the one with the largest Q."""
    agent = QLearningAgent()
    board = [1, -1, 0, 0, 0, 0, 0, 0, 0]
    agent.q_table[_key(board)] = {"2": 0.1, "4": 0.9, "8": 0.3}
    obs = np.array(board, dtype=np.int8).reshape(3, 3)
    legal = [i for i, v in enumerate(board) if v == 0]
    assert agent.act(obs, {"legal_actions": legal}) == 4


def test_act_uses_d4_symmetry_for_rotated_board():
    """A rotated board reuses the stored entry, and the chosen move rotates with it."""
    agent = QLearningAgent()
    stored = np.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
    agent.q_table[_key(stored.flatten().tolist())] = {"5": 1.0}

    rotated = np.rot90(stored)
    origin_of_cell = np.rot90(np.arange(9).reshape(3, 3)).flatten()
    expected_action = int(np.flatnonzero(origin_of_cell == 5)[0])

    legal = [i for i, v in enumerate(rotated.flatten()) if v == 0]
    assert agent.act(rotated, {"legal_actions": legal}) == expected_action


# ── Learning (Bellman update) ────────────────────────────────────────────────


def test_learn_terminal_step_moves_q_toward_reward():
    """At game end the target is just the reward: Q <- Q + alpha * (r - Q)."""
    agent = QLearningAgent(alpha=0.1)
    old_obs = np.zeros((3, 3), dtype=np.int8)
    agent.learn(old_obs, 4, 1.0, old_obs, {"legal_actions": []}, done=True)
    assert agent.q_table[_key([0] * 9)]["4"] == pytest.approx(0.1)


def test_learn_non_terminal_step_bootstraps_from_best_next_q():
    """Mid-game the target is r + gamma * max_a' Q(s', a')."""
    agent = QLearningAgent(alpha=0.1, gamma=0.9)
    old_obs = np.zeros((3, 3), dtype=np.int8)
    next_board = [1, -1, 0, 0, 0, 0, 0, 0, 0]
    agent.q_table[_key(next_board)] = {"2": 0.5, "3": 0.2}
    next_obs = np.array(next_board, dtype=np.int8).reshape(3, 3)

    agent.learn(old_obs, 0, 0.0, next_obs, {"legal_actions": [2, 3]}, done=False)

    assert agent.q_table[_key([0] * 9)]["0"] == pytest.approx(0.1 * 0.9 * 0.5)


# ── Persistence ──────────────────────────────────────────────────────────────


def test_save_and_load_round_trip(tmp_path: Path):
    """A saved table loads back identically."""
    agent = QLearningAgent()
    agent.q_table = {_key([0] * 9): {"4": 0.7, "0": 0.2}}
    path = tmp_path / "q.json"
    agent.save_model(str(path))

    restored = QLearningAgent()
    restored.load_model(str(path))
    assert restored.q_table == agent.q_table


# ── Trained model regression ─────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_trained_table_never_loses_to_alphabeta(seed: int):
    """The committed Q-table must hold the 'unbeatable' claim on both sides."""
    random.seed(seed)
    q_as_x = QLearningAgent(player=1)
    q_as_x.load_model(str(TRAINED_Q_TABLE))
    q_as_o = QLearningAgent(player=-1)
    q_as_o.load_model(str(TRAINED_Q_TABLE))

    assert _play_one_game(q_as_x, AlphaBetaAgent(player=-1)) != -1
    assert _play_one_game(AlphaBetaAgent(player=1), q_as_o) != 1
