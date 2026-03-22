"""
Tests for Tic-Tac-Toe Game Engine.
"""

from board_game_rl.games.tic_tac_toe.engine import TicTacToeEngine


def test_engine_initialization():
    engine = TicTacToeEngine()
    assert engine.current_player == 1
    assert engine.winner is None
    assert len(engine.get_legal_actions()) == 9


def test_engine_valid_moves():
    engine = TicTacToeEngine()

    # Player 1 moves to center
    success = engine.step((1, 1))
    assert success is True
    assert engine.board[1][1] == 1
    assert engine.current_player == -1

    # Player -1 moves to top-left
    success = engine.step((0, 0))
    assert success is True
    assert engine.board[0][0] == -1
    assert engine.current_player == 1


def test_engine_invalid_moves():
    engine = TicTacToeEngine()
    engine.step((1, 1))

    # Player -1 tries to move to an occupied space
    success = engine.step((1, 1))
    assert success is False
    assert engine.board[1][1] == 1  # Should not change
    assert engine.current_player == -1  # Turn shouldn't change


def test_engine_win_condition_row():
    engine = TicTacToeEngine()

    # Row win for player 1
    engine.step((0, 0))  # P1
    engine.step((1, 0))  # P2
    engine.step((0, 1))  # P1
    engine.step((1, 1))  # P2
    engine.step((0, 2))  # P1

    assert engine.winner == 1


def test_engine_draw_condition():
    engine = TicTacToeEngine()
    moves = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)]
    for move in moves:
        engine.step(move)

    assert engine.winner == 0
    assert len(engine.get_legal_actions()) == 0


def test_engine_step_after_game_over():
    engine = TicTacToeEngine()
    # Win game
    engine.step((0, 0))  # P1
    engine.step((1, 0))  # P2
    engine.step((0, 1))  # P1
    engine.step((1, 1))  # P2
    engine.step((0, 2))  # P1

    # Try to step after game over
    success = engine.step((2, 2))
    assert success is False
