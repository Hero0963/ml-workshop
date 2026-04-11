# tests/api/test_inference.py


from board_game_rl.api.inference import _agent_cache, _get_agent, get_optimal_move


def test_agent_cache_returns_same_instance():
    """Verify _get_agent returns the same cached instance on repeated calls."""
    _agent_cache.clear()
    agent_1 = _get_agent("Q-Learning", player=1)
    agent_2 = _get_agent("Q-Learning", player=1)
    assert agent_1 is agent_2


def test_agent_cache_separates_by_player():
    """Verify _get_agent caches separately per player."""
    _agent_cache.clear()
    agent_x = _get_agent("Q-Learning", player=1)
    agent_o = _get_agent("Q-Learning", player=-1)
    assert agent_x is not agent_o
    assert agent_x.player == 1
    assert agent_o.player == -1


def test_get_optimal_move_returns_valid_action():
    """Verify get_optimal_move returns a legal action index on an empty board."""
    empty_board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for agent_type in ["Alpha-Beta", "Q-Learning", "Random"]:
        action = get_optimal_move(empty_board, current_player=1, agent_type=agent_type)
        assert 0 <= action <= 8


def test_get_optimal_move_no_legal_actions():
    """Verify get_optimal_move returns -1 when the board is full."""
    full_board = [[1, -1, 1], [-1, 1, -1], [-1, 1, -1]]
    action = get_optimal_move(full_board, current_player=1)
    assert action == -1
