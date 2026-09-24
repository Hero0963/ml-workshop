# tests/games/tic_tac_toe/test_rules.py

import random

import numpy as np
import pytest

from board_game_rl.games.tic_tac_toe.engine import TicTacToeEngine
from board_game_rl.games.tic_tac_toe.rules import TicTacToeRules, TicTacToeState

RULES = TicTacToeRules()
DIFFERENTIAL_GAMES = 1_000
DIFFERENTIAL_SEED = 20260924


def _state(board: list[int], to_move: int) -> TicTacToeState:
    return TicTacToeState(board=tuple(board), to_move=to_move)


def test_initial_state_is_empty_with_x_to_move():
    state = RULES.initial_state()
    assert state.board == (0,) * 9
    assert RULES.to_move(state) == 1
    assert RULES.legal_actions(state) == list(range(9))
    assert RULES.winner(state) is None


def test_next_state_places_stone_switches_player_and_keeps_original():
    start = RULES.initial_state()
    after = RULES.next_state(start, 4)
    assert after.board[4] == 1
    assert RULES.to_move(after) == -1
    assert start.board == (0,) * 9


def test_next_state_rejects_occupied_cell():
    state = RULES.next_state(RULES.initial_state(), 4)
    with pytest.raises(ValueError):
        RULES.next_state(state, 4)


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        ([1, 1, 1, -1, -1, 0, 0, 0, 0], 1),  # row
        ([-1, 1, 0, -1, 1, 0, -1, 0, 1], -1),  # column
        ([1, -1, 0, -1, 1, 0, 0, 0, 1], 1),  # main diagonal
        ([1, 1, -1, 0, -1, 1, -1, 0, 0], -1),  # anti-diagonal
        ([1, -1, 1, 1, -1, -1, -1, 1, 1], 0),  # full board, no line
        ([1, -1, 0, 0, 0, 0, 0, 0, 0], None),  # still playing
    ],
)
def test_winner(board: list[int], expected: int | None):
    assert RULES.winner(_state(board, to_move=1)) == expected


def test_from_observation_flattens_3x3_board():
    observation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.int8)
    state = RULES.from_observation(observation, to_move=1)
    assert state == _state([1, 0, 0, 0, -1, 0, 0, 0, 0], to_move=1)
    assert all(type(value) is int for value in state.board)


def test_rules_agree_with_engine_on_random_games():
    """Differential test: the new rules must match the engine move by move."""
    rng = random.Random(DIFFERENTIAL_SEED)
    for _ in range(DIFFERENTIAL_GAMES):
        engine = TicTacToeEngine()
        state = RULES.initial_state()
        while engine.winner is None:
            engine_legal = [r * 3 + c for r, c in engine.get_legal_actions()]
            assert RULES.legal_actions(state) == engine_legal
            assert RULES.to_move(state) == engine.current_player

            action = rng.choice(engine_legal)
            engine.step((action // 3, action % 3))
            state = RULES.next_state(state, action)

            assert RULES.winner(state) == engine.winner
            assert list(state.board) == [v for row in engine.board for v in row]
