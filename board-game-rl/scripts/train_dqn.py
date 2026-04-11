# scripts/train_dqn.py
"""
Train DQN Agent for Tic-Tac-Toe.

Uses the same mixed-opponent strategy as Q-Learning training:
Random / Self-Play / Alpha-Beta / Hybrid.

Usage (from board-game-rl/):
    uv run python scripts/train_dqn.py
"""

import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


sys.path.append(str(Path(__file__).parent.parent / "src"))

from board_game_rl.agents.dqn_agent import DQNAgent
from board_game_rl.agents.random_agent import RandomAgent
from board_game_rl.games.tic_tac_toe.env import TicTacToeEnv


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    total_episodes: int = 100_000
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_interval: int = 500
    epsilon_decay_rate: float = 0.95
    batch_size: int = 64
    buffer_capacity: int = 50_000
    target_update_freq: int = 500
    checkpoint_interval: int = 1_000
    log_interval: int = 5_000
    validation_interval: int = 10_000
    quick_validation_games: int = 500
    draw_reward: float = 0.5
    win_reward: float = 1.0
    loss_reward: float = -1.0
    max_random_opponent_moves: int = 3
    validation_games_per_side: int = 5_000


# ── Cached Alpha-Beta Opponent ───────────────────────────────────────────────


def load_alphabeta_cache(cache_path: Path) -> dict[str, int]:
    with open(cache_path) as f:
        return json.load(f)


def cached_ab_act(cache: dict[str, int], observation: np.ndarray) -> int:
    state_key = str(observation.flatten().tolist())
    return cache[state_key]


# ── Training Core ────────────────────────────────────────────────────────────


@dataclass
class Checkpoint:
    episode: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    avg_loss: float
    epsilon: float


def _get_final_reward(
    winner: int, q_player: int, config: TrainConfig
) -> tuple[float, int]:
    if winner == q_player:
        return config.win_reward, 1
    elif winner == 0:
        return config.draw_reward, 0
    else:
        return config.loss_reward, -1


OPPONENT_TYPES = ["random", "self_play", "alphabeta", "hybrid"]


def _train_episode_self_play(
    env: TicTacToeEnv,
    agent: DQNAgent,
    q_player: int,
    config: TrainConfig,
) -> tuple[int, list[float]]:
    """Self-play: both sides use the DQN agent. Returns (result, losses)."""
    observation, info = env.reset()
    terminated = False
    last: dict[int, tuple] = {1: (None, None, None), -1: (None, None, None)}
    losses: list[float] = []

    while not terminated:
        cp = info["current_player"]
        agent.player = cp

        prev_obs, prev_action, prev_legal = last[cp]
        if prev_obs is not None:
            loss = agent.learn(
                prev_obs,
                prev_action,
                0.0,
                observation,
                False,
                info.get("legal_actions", []),
            )
            if loss is not None:
                losses.append(loss)

        action = agent.act(observation, info, is_training=True)
        last[cp] = (observation.copy(), action, info.get("legal_actions", []))
        observation, _, terminated, _, info = env.step(action)

        if terminated:
            winner = env.unwrapped.engine.winner
            for player in [1, -1]:
                p_obs, p_action, _p_legal = last[player]
                if p_obs is not None:
                    agent.player = player
                    reward, _ = _get_final_reward(winner, player, config)
                    loss = agent.learn(p_obs, p_action, reward, observation, True, [])
                    if loss is not None:
                        losses.append(loss)

            result = 1 if winner == q_player else (0 if winner == 0 else -1)
            return result, losses

    return 0, losses


def train_episode(
    env: TicTacToeEnv,
    agent: DQNAgent,
    cache: dict[str, int],
    q_player: int,
    config: TrainConfig,
) -> tuple[int, list[float]]:
    """Run one training episode. Returns (result, losses)."""
    opponent = random.choice(OPPONENT_TYPES)

    if opponent == "self_play":
        return _train_episode_self_play(env, agent, q_player, config)

    agent.player = q_player
    observation, info = env.reset()
    terminated = False
    last_obs, last_action = None, None
    losses: list[float] = []

    if opponent == "random":
        random_moves_left = 999
    elif opponent == "alphabeta":
        random_moves_left = 0
    else:  # hybrid
        random_moves_left = random.randint(1, config.max_random_opponent_moves)

    while not terminated:
        current_player = info["current_player"]

        if current_player == q_player:
            if last_obs is not None:
                loss = agent.learn(
                    last_obs,
                    last_action,
                    0.0,
                    observation,
                    False,
                    info.get("legal_actions", []),
                )
                if loss is not None:
                    losses.append(loss)

            action = agent.act(observation, info, is_training=True)
            last_obs = observation.copy()
            last_action = action
            observation, _, terminated, _, info = env.step(action)

            if terminated:
                winner = env.unwrapped.engine.winner
                reward, result = _get_final_reward(winner, q_player, config)
                loss = agent.learn(last_obs, last_action, reward, observation, True, [])
                if loss is not None:
                    losses.append(loss)
                return result, losses
        else:
            if random_moves_left > 0:
                legal = info.get("legal_actions", [])
                action = random.choice(legal)
                random_moves_left -= 1
            else:
                action = cached_ab_act(cache, observation)

            observation, _, terminated, _, info = env.step(action)

            if terminated and last_obs is not None:
                winner = env.unwrapped.engine.winner
                reward, result = _get_final_reward(winner, q_player, config)
                loss = agent.learn(last_obs, last_action, reward, observation, True, [])
                if loss is not None:
                    losses.append(loss)
                return result, losses

    return 0, losses


# ── Validation ───────────────────────────────────────────────────────────────


def validate(
    agent: DQNAgent,
    opponent_act_fn,
    q_player: int,
    num_games: int,
) -> tuple[int, int, int]:
    """Play games with no learning/exploration. Returns (wins, draws, losses)."""
    env = TicTacToeEnv()
    agent.player = q_player
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins, draws, losses = 0, 0, 0

    for _ in range(num_games):
        observation, info = env.reset()
        terminated = False

        while not terminated:
            if info["current_player"] == q_player:
                action = agent.act(observation, info, is_training=False)
            else:
                action = opponent_act_fn(observation, info)
            observation, _, terminated, _, info = env.step(action)

        winner = env.unwrapped.engine.winner
        if winner == q_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    agent.epsilon = original_epsilon
    return wins, draws, losses


def quick_validate_vs_ab(
    agent: DQNAgent,
    cache: dict[str, int],
    games_per_side: int,
) -> tuple[int, dict[str, tuple[int, int, int]]]:
    """Quick validation vs Alpha-Beta only. Returns (total_losses, results)."""

    def ab_fn(obs, _info):
        return cached_ab_act(cache, obs)

    results = {}
    total_losses = 0
    for q_player, side in [(1, "X"), (-1, "O")]:
        w, d, lo = validate(agent, ab_fn, q_player, games_per_side)
        results[f"vs_AB_as_{side}"] = (w, d, lo)
        total_losses += lo
    return total_losses, results


def run_full_validation(
    agent: DQNAgent,
    cache: dict[str, int],
    games_per_side: int,
) -> dict:
    """Validate against Alpha-Beta, Random, and Self from both sides."""
    random_agent = RandomAgent()
    results = {}

    def ab_fn(obs, _info):
        return cached_ab_act(cache, obs)

    def rand_fn(obs, info):
        return random_agent.act(obs, info)

    for q_player, side in [(1, "X"), (-1, "O")]:
        w, d, lo = validate(agent, ab_fn, q_player, games_per_side)
        results[f"vs_AB_as_{side}"] = (w, d, lo)
        status = "PASS" if lo == 0 else f"FAIL({lo})"
        print(f"    vs Alpha-Beta (as {side}): W={w:,} D={d:,} L={lo:,} {status}")

    for q_player, side in [(1, "X"), (-1, "O")]:
        w, d, lo = validate(agent, rand_fn, q_player, games_per_side)
        results[f"vs_Rand_as_{side}"] = (w, d, lo)
        status = "PASS" if lo == 0 else f"FAIL({lo})"
        print(f"    vs Random    (as {side}): W={w:,} D={d:,} L={lo:,} {status}")

    # Self-play with a copy
    for q_player, side in [(1, "X"), (-1, "O")]:
        opp = DQNAgent(player=-q_player, epsilon=0.0, device=str(agent.device))
        opp.policy_net.load_state_dict(agent.policy_net.state_dict())
        opp.sync_target()

        def self_fn(obs, info, _a=opp):
            return _a.act(obs, info, is_training=False)

        w, d, lo = validate(agent, self_fn, q_player, games_per_side)
        results[f"vs_Self_as_{side}"] = (w, d, lo)
        status = "PASS" if lo == 0 else f"FAIL({lo})"
        print(f"    vs Self      (as {side}): W={w:,} D={d:,} L={lo:,} {status}")

    return results


# ── Report ───────────────────────────────────────────────────────────────────


def generate_report(
    config: TrainConfig,
    checkpoints: list[Checkpoint],
    elapsed: float,
    val_results: dict,
    device: str,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")

    step = max(1, len(checkpoints) // 25)
    sampled = checkpoints[::step]
    if sampled and sampled[-1].episode != checkpoints[-1].episode:
        sampled.append(checkpoints[-1])

    chart_x = json.dumps([cp.episode // 1000 for cp in sampled])
    chart_draw = json.dumps([round(cp.draw_rate, 1) for cp in sampled])
    chart_loss = json.dumps([round(cp.loss_rate, 1) for cp in sampled])
    chart_nn_loss = json.dumps([round(cp.avg_loss, 4) for cp in sampled])

    conv_rows = "\n".join(
        f"| {cp.episode:,} | {cp.win_rate:.1f}% | {cp.draw_rate:.1f}% "
        f"| {cp.loss_rate:.1f}% | {cp.avg_loss:.4f} | {cp.epsilon:.4f} |"
        for cp in checkpoints
    )

    def val_rows(val: dict) -> str:
        opp_names = {"AB": "Alpha-Beta", "Rand": "Random", "Self": "Self"}
        lines = []
        for key, (w, d, lo) in val.items():
            parts = key.split("_")
            opp = opp_names.get(parts[1], parts[1])
            side = parts[3]
            status = "PASS" if lo == 0 else "FAIL"
            lines.append(
                f"| vs {opp} (as {side}) | {w:,} | {d:,} | {lo:,} | {status} |"
            )
        return "\n".join(lines)

    all_losses = sum(lo for _, (_, _, lo) in val_results.items())
    verdict = "UNBEATABLE" if all_losses == 0 else "NOT YET CONVERGED"

    return f"""# DQN Training Report

> **Date**: _{today}_
> **Goal**: Train a Deep Q-Network agent for Tic-Tac-Toe
> **Device**: {device}

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Total Episodes | {config.total_episodes:,} |
| Learning Rate | {config.lr} |
| Discount Factor (gamma) | {config.gamma} |
| Epsilon (start → end) | {config.epsilon_start} → {config.epsilon_end} |
| Batch Size | {config.batch_size} |
| Buffer Capacity | {config.buffer_capacity:,} |
| Target Update Freq | Every {config.target_update_freq} learn steps |
| Network | MLP: 9 → 128 → 128 → 9 |
| Opponent | Mixed: Random / Self-Play / Alpha-Beta / Hybrid |

---

## Training Curves

### Win/Draw/Loss Rate

```mermaid
xychart-beta
    title "Training Convergence (per {config.checkpoint_interval:,}-episode window)"
    x-axis "Episodes (x1000)" {chart_x}
    y-axis "Rate (%)" 0 --> 100
    line "Draw Rate" {chart_draw}
    line "Loss Rate" {chart_loss}
```

### Neural Network Loss

```mermaid
xychart-beta
    title "Average Training Loss"
    x-axis "Episodes (x1000)" {chart_x}
    y-axis "MSE Loss"
    line "Avg Loss" {chart_nn_loss}
```

### Detailed Convergence Data

| Episode | Win | Draw | Loss | Avg Loss | Epsilon |
|---------|-----|------|------|----------|---------|
{conv_rows}

---

## Validation Results ({config.validation_games_per_side:,} games per side)

| Match | Win | Draw | Loss | Status |
|-------|-----|------|------|--------|
{val_rows(val_results)}

---

## Result: **{verdict}**

{'The DQN agent achieves **zero losses** against Alpha-Beta, Random, and itself from both sides.' if all_losses == 0 else f'The agent has {all_losses} total losses. Neural network approximation may require more training or hyperparameter tuning compared to tabular Q-Learning.'}

## DQN vs Tabular Q-Learning

| Aspect | Tabular Q-Learning | DQN |
|--------|-------------------|-----|
| State Representation | Exact lookup (3,441 states) | Neural network generalization |
| Scalability | Limited to small state spaces | Can scale to larger games |
| Training Time | ~{elapsed:.0f}s | {elapsed:.1f}s |
| Optimality | Guaranteed (with enough exploration) | Approximate |
"""


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    project_root = Path(__file__).parent.parent
    cache_path = project_root / "models" / "alphabeta_cache.json"
    model_path = project_root / "models" / "dqn_model.pth"
    report_dir = project_root / "ai-collab" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = report_dir / f"dqn_training_report_{today}.md"

    config = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(" DQN Training: Tic-Tac-Toe")
    print("=" * 60)
    print(f"  Episodes:     {config.total_episodes:,}")
    print(f"  LR:           {config.lr}  |  Gamma: {config.gamma}")
    print(f"  Epsilon:      {config.epsilon_start} → {config.epsilon_end}")
    print(f"  Batch Size:   {config.batch_size}  |  Buffer: {config.buffer_capacity:,}")
    print(f"  Target Sync:  Every {config.target_update_freq} learn steps")
    print(
        f"  Validate:     Every {config.validation_interval:,} eps "
        f"({config.quick_validation_games} games/side vs AB)"
    )
    print(f"  Device:       {device}")
    print()

    if not cache_path.exists():
        print(f"ERROR: Alpha-Beta cache not found at {cache_path}")
        print("Run `uv run python scripts/precompute_alphabeta.py` first.")
        sys.exit(1)

    cache = load_alphabeta_cache(cache_path)
    print(f"  Loaded Alpha-Beta cache: {len(cache):,} states")
    print()

    # ── Training with live logging ───────────────────────────────────────
    total = config.total_episodes
    ep_width = len(f"{total:,}")

    print("-" * 60)
    print(
        " Training  (log every {:,} eps | validate every {:,} eps)".format(
            config.log_interval, config.validation_interval
        )
    )
    print("-" * 60)

    agent = DQNAgent(
        lr=config.lr,
        gamma=config.gamma,
        epsilon=config.epsilon_start,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
        buffer_capacity=config.buffer_capacity,
        device=device,
    )

    env = TicTacToeEnv()
    checkpoints: list[Checkpoint] = []
    window_w, window_d, window_l = 0, 0, 0
    window_losses: list[float] = []
    best_ab_losses = float("inf")
    best_saved_at = 0

    start = time.perf_counter()

    for episode in range(total):
        q_player = random.choice([1, -1])
        result, ep_losses = train_episode(env, agent, cache, q_player, config)
        window_losses.extend(ep_losses)

        if result == 1:
            window_w += 1
        elif result == 0:
            window_d += 1
        else:
            window_l += 1

        # Checkpoint (record metrics)
        if (episode + 1) % config.checkpoint_interval == 0:
            w_total = window_w + window_d + window_l
            avg_loss = sum(window_losses) / len(window_losses) if window_losses else 0.0
            checkpoints.append(
                Checkpoint(
                    episode=episode + 1,
                    win_rate=window_w / w_total * 100,
                    draw_rate=window_d / w_total * 100,
                    loss_rate=window_l / w_total * 100,
                    avg_loss=avg_loss,
                    epsilon=agent.epsilon,
                )
            )
            window_w, window_d, window_l = 0, 0, 0
            window_losses.clear()

        # Live log
        if (episode + 1) % config.log_interval == 0:
            cp = checkpoints[-1] if checkpoints else None
            elapsed = time.perf_counter() - start
            if cp:
                print(
                    f"  [{episode + 1:>{ep_width},}/{total:,}]"
                    f"  W:{cp.win_rate:5.1f}%  D:{cp.draw_rate:5.1f}%  L:{cp.loss_rate:5.1f}%"
                    f"  loss:{cp.avg_loss:.4f}  eps:{agent.epsilon:.4f}"
                    f"  ({elapsed:.1f}s)"
                )

        # Periodic validation vs Alpha-Beta + best model save
        if (episode + 1) % config.validation_interval == 0:
            ab_losses, ab_results = quick_validate_vs_ab(
                agent, cache, config.quick_validation_games
            )
            ab_x = ab_results["vs_AB_as_X"]
            ab_o = ab_results["vs_AB_as_O"]
            print(
                f"    >>> Validate vs AB:"
                f"  X(W:{ab_x[0]} D:{ab_x[1]} L:{ab_x[2]})"
                f"  O(W:{ab_o[0]} D:{ab_o[1]} L:{ab_o[2]})"
                f"  Total L:{ab_losses}"
            )

            if ab_losses <= best_ab_losses:
                best_ab_losses = ab_losses
                best_saved_at = episode + 1
                agent.save_model(str(model_path))
                print(
                    f"    >>> Best model saved! (losses: {ab_losses}"
                    f"/{config.quick_validation_games * 2})"
                )

        # Epsilon decay
        if (episode + 1) % config.epsilon_decay_interval == 0:
            agent.epsilon = max(
                config.epsilon_end,
                agent.epsilon * config.epsilon_decay_rate,
            )

    elapsed = time.perf_counter() - start
    print()
    print(f"  Training done in {elapsed:.1f}s")
    print(f"  Learn steps:    {agent.learn_step_count:,}")
    print(f"  Buffer size:    {len(agent.replay_buffer):,}")
    print(f"  Best model at:  episode {best_saved_at:,} (AB losses: {best_ab_losses})")
    print()

    # ── Final full validation (using best saved model) ───────────────────
    print("-" * 60)
    print(f" Final Validation ({config.validation_games_per_side:,} games/side)")
    print("-" * 60)

    # Reload the best model for final validation
    if model_path.exists():
        agent.load_model(str(model_path))

    val_results = run_full_validation(agent, cache, config.validation_games_per_side)
    print()

    # ── Report ───────────────────────────────────────────────────────────
    report = generate_report(config, checkpoints, elapsed, val_results, device)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    total_losses = sum(lo for _, (_, _, lo) in val_results.items())
    print()
    print("=" * 60)
    if total_losses == 0:
        print(" RESULT: UNBEATABLE (0 losses vs AB, Random, and Self)")
    else:
        print(f" RESULT: {total_losses} total losses remain")
        print(f"         Best checkpoint was at episode {best_saved_at:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
