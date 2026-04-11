# tests/agents/test_dqn_agent.py

import numpy as np
import torch

from board_game_rl.agents.dqn_agent import DQNAgent, DQNNetwork, ReplayBuffer


# ── DQNNetwork ───────────────────────────────────────────────────────────────


def test_network_output_shape():
    """Network should output 9 Q-values for a single board input."""
    net = DQNNetwork()
    x = torch.zeros(1, 9)
    out = net(x)
    assert out.shape == (1, 9)


def test_network_batch_forward():
    """Network should handle batch inputs."""
    net = DQNNetwork()
    x = torch.randn(32, 9)
    out = net(x)
    assert out.shape == (32, 9)


# ── ReplayBuffer ─────────────────────────────────────────────────────────────


def test_replay_buffer_push_and_len():
    """Buffer should track stored transitions."""
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0

    buf.push(np.zeros(9), 0, 1.0, np.zeros(9), False, [1, 2, 3])
    assert len(buf) == 1


def test_replay_buffer_capacity():
    """Buffer should not exceed capacity."""
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(np.zeros(9), i % 9, 0.0, np.zeros(9), False, [0])
    assert len(buf) == 5


def test_replay_buffer_sample():
    """Sampled batch should have correct size."""
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.push(np.zeros(9), i % 9, 0.0, np.zeros(9), False, [0])
    batch = buf.sample(10)
    assert len(batch) == 10


# ── DQNAgent ─────────────────────────────────────────────────────────────────


def test_agent_act_returns_legal_action():
    """Agent should always return a legal action."""
    agent = DQNAgent(device="cpu")
    board = np.zeros((3, 3), dtype=np.int8)
    info = {"legal_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8]}
    action = agent.act(board, info, is_training=False)
    assert action in info["legal_actions"]


def test_agent_act_respects_legal_mask():
    """Agent should never pick an illegal action."""
    agent = DQNAgent(epsilon=0.0, device="cpu")
    board = np.array([[1, -1, 1], [-1, 0, 0], [1, -1, 0]], dtype=np.int8)
    legal = [4, 5, 8]
    info = {"legal_actions": legal}

    for _ in range(50):
        action = agent.act(board, info, is_training=False)
        assert action in legal


def test_agent_learn_returns_none_when_buffer_small():
    """Learn should return None when buffer < batch_size."""
    agent = DQNAgent(batch_size=64, device="cpu")
    loss = agent.learn(np.zeros((3, 3)), 0, 1.0, np.zeros((3, 3)), True, [])
    assert loss is None


def test_agent_learn_returns_loss():
    """Learn should return a float loss after enough transitions."""
    agent = DQNAgent(batch_size=8, device="cpu")

    for i in range(16):
        agent.learn(
            np.random.randn(3, 3).astype(np.float32),
            i % 9,
            1.0 if i % 3 == 0 else 0.0,
            np.random.randn(3, 3).astype(np.float32),
            i % 5 == 0,
            [j for j in range(9) if j != i % 9],
        )

    loss = agent.learn(
        np.random.randn(3, 3).astype(np.float32),
        0,
        1.0,
        np.random.randn(3, 3).astype(np.float32),
        True,
        [],
    )
    assert isinstance(loss, float)
    assert loss >= 0


def test_agent_board_normalization():
    """Player -1 should see a flipped board."""
    agent_x = DQNAgent(player=1, device="cpu")
    agent_o = DQNAgent(player=-1, device="cpu")

    board = np.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)

    norm_x = agent_x._normalize_obs(board)
    norm_o = agent_o._normalize_obs(board)

    np.testing.assert_array_equal(norm_x, board)
    np.testing.assert_array_equal(norm_o, board * -1)


def test_agent_save_and_load(tmp_path):
    """Saved model should produce identical Q-values after loading."""
    agent = DQNAgent(device="cpu")
    board = np.zeros((3, 3), dtype=np.int8)

    with torch.no_grad():
        q_before = agent.policy_net(agent._obs_to_tensor(board)).clone()

    model_path = str(tmp_path / "dqn_test.pth")
    agent.save_model(model_path)

    agent2 = DQNAgent(device="cpu")
    agent2.load_model(model_path)

    with torch.no_grad():
        q_after = agent2.policy_net(agent2._obs_to_tensor(board))

    torch.testing.assert_close(q_before, q_after)


def test_agent_target_sync():
    """Target network should match policy network after sync."""
    agent = DQNAgent(device="cpu")

    for p_param, t_param in zip(
        agent.policy_net.parameters(), agent.target_net.parameters()
    ):
        torch.testing.assert_close(p_param, t_param)
