# tests/agents/test_mcts_agent.py

import logging

import numpy as np
import pytest

from board_game_rl.agents.mcts_agent import MCTSAgent
from board_game_rl.games.tic_tac_toe.alphabeta_agent import AlphaBetaAgent
from board_game_rl.games.tic_tac_toe.env import TicTacToeEnv
from board_game_rl.games.tic_tac_toe.rules import TicTacToeRules, TicTacToeState

RULES = TicTacToeRules()
SEEDS = [0, 1, 2]
TACTICS_SIMULATIONS = 500


def _state(board: list[int], to_move: int) -> TicTacToeState:
    return TicTacToeState(board=tuple(board), to_move=to_move)


def _agent(n_simulations: int, seed: int, player: int = 1) -> MCTSAgent:
    return MCTSAgent(RULES, player=player, n_simulations=n_simulations, seed=seed)


# ── Tree bookkeeping ─────────────────────────────────────────────────────────


def test_returns_legal_action_on_empty_board():
    action = _agent(n_simulations=50, seed=0).search(RULES.initial_state())
    assert action in range(9)


def test_root_visits_equal_simulations_and_split_over_children():
    agent = _agent(n_simulations=300, seed=0)
    agent.search(RULES.initial_state())
    root = agent.last_root
    assert root.visits == 300
    assert sum(child.visits for child in root.children) == 300


def test_scores_stay_within_zero_and_one():
    agent = _agent(n_simulations=300, seed=0)
    agent.search(RULES.initial_state())
    for child in agent.last_root.children:
        assert 0.0 <= child.mean_score <= 1.0


def test_same_seed_gives_same_search():
    first, second = _agent(200, seed=7), _agent(200, seed=7)
    assert first.search(RULES.initial_state()) == second.search(RULES.initial_state())
    assert [c.visits for c in first.last_root.children] == [
        c.visits for c in second.last_root.children
    ]


def test_single_legal_action_is_returned_without_search():
    state = _state([1, -1, 1, 1, -1, -1, -1, 1, 0], to_move=1)
    agent = _agent(n_simulations=10, seed=0)
    assert agent.search(state) == 8
    assert agent.last_root is None


def test_rejects_finished_game():
    with pytest.raises(ValueError):
        _agent(10, seed=0).search(_state([1, 1, 1, -1, -1, 0, 0, 0, 0], to_move=-1))


def test_rejects_non_positive_simulations():
    with pytest.raises(ValueError):
        MCTSAgent(RULES, n_simulations=0)


# ── Tactics: these fail if backpropagation scores the wrong player ───────────


@pytest.mark.parametrize("seed", SEEDS)
def test_takes_immediate_win(seed: int):
    """X completes the top row instead of anything else (O also threatens 5)."""
    state = _state([1, 1, 0, -1, -1, 0, 0, 0, 0], to_move=1)
    assert _agent(TACTICS_SIMULATIONS, seed).search(state) == 2


@pytest.mark.parametrize("seed", SEEDS)
def test_blocks_opponent_immediate_win(seed: int):
    """O has no win of its own, so it must block X's top row at 2."""
    state = _state([1, 1, 0, 0, -1, 0, 0, 0, 0], to_move=-1)
    assert _agent(TACTICS_SIMULATIONS, seed, player=-1).search(state) == 2


def test_act_reads_board_from_observation_as_its_own_player():
    observation = np.array([[1, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.int8)
    agent = _agent(TACTICS_SIMULATIONS, seed=0, player=-1)
    assert agent.act(observation, {"legal_actions": [2, 3, 5, 6, 7, 8]}) == 2


# ── Against the perfect player ───────────────────────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_does_not_lose_to_alphabeta(seed: int):
    logging.disable(logging.INFO)
    try:
        for mcts_player in (1, -1):
            env = TicTacToeEnv()
            obs, info = env.reset()
            agents = {
                mcts_player: _agent(1_000, seed, player=mcts_player),
                -mcts_player: AlphaBetaAgent(player=-mcts_player),
            }
            terminated = False
            while not terminated:
                action = agents[info["current_player"]].act(obs, info)
                obs, _, terminated, _, info = env.step(action)
            assert env.engine.winner != -mcts_player
    finally:
        logging.disable(logging.NOTSET)
