import numpy as np

from board_game_rl.agents.alphabeta_agent import AlphaBetaAgent


def test_alphabeta_first_move():
    """Test that AlphaBetaAgent can choose a move on an empty board without crashing."""
    agent = AlphaBetaAgent(player=1)

    # Empty board
    empty_board = np.zeros((3, 3), dtype=int)

    # Act
    action = agent.act(empty_board)
    assert 0 <= action <= 8


def test_alphabeta_winning_move():
    """Test that AlphaBetaAgent picks the winning move if available."""
    agent = AlphaBetaAgent(player=1)

    # Board where X (1) can win by taking (0, 2)
    board = np.array([[1, 1, 0], [-1, -1, 0], [0, 0, 0]])

    action = agent.act(board)
    assert action == 2  # (0, 2) is index 2


def test_alphabeta_blocking_move():
    """Test that AlphaBetaAgent blocks opponent from winning."""
    agent = AlphaBetaAgent(player=1)  # AI is X (1)

    # Board where O (-1) is about to win if X doesn't block (2,2)
    board = np.array([[-1, 0, 0], [0, -1, 0], [1, 1, 0]])

    action = agent.act(board)
    # X can either block O at (2,2) or win instantly at (2,2). Both are the same move: 8.
    assert action == 8
