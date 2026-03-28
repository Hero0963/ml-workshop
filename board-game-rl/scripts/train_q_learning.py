# scripts/train_q_learning.py
"""
Train Q-Learning Agent to achieve unbeatable play in Tic-Tac-Toe.

Trains against Alpha-Beta (perfect opponent) with board normalization,
so a single Q-table works for both X and O.

Usage (from board-game-rl/):
    uv run python scripts/train_q_learning.py
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from board_game_rl.agents.q_learning_agent import QLearningAgent
from board_game_rl.agents.random_agent import RandomAgent
from board_game_rl.games.tic_tac_toe.env import TicTacToeEnv


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    total_episodes: int = 150_000
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 0.3
    epsilon_end: float = 0.01
    epsilon_decay_interval: int = 1000
    epsilon_decay_rate: float = 0.9
    checkpoint_interval: int = 1000
    draw_reward: float = 0.5
    win_reward: float = 1.0
    loss_reward: float = -1.0
    validation_games_per_side: int = 5_000
    max_random_opponent_moves: int = 3  # Random prefix before AB takeover


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
    q_table_size: int
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
    q_agent: QLearningAgent,
    q_player: int,
    config: TrainConfig,
) -> int:
    """Both sides use Q-agent. Returns result from q_player's perspective."""
    observation, info = env.reset()
    terminated = False
    last: dict[int, tuple] = {1: (None, None), -1: (None, None)}

    while not terminated:
        cp = info["current_player"]
        q_agent.player = cp

        prev_obs, prev_action = last[cp]
        if prev_obs is not None:
            q_agent.learn(prev_obs, prev_action, 0.0, observation, info, False)

        action = q_agent.act(observation, info, is_training=True)
        last[cp] = (observation.copy(), action)
        observation, _, terminated, _, info = env.step(action)

        if terminated:
            winner = env.unwrapped.engine.winner
            for player in [1, -1]:
                p_obs, p_action = last[player]
                if p_obs is not None:
                    q_agent.player = player
                    reward, _ = _get_final_reward(winner, player, config)
                    q_agent.learn(p_obs, p_action, reward, observation, info, True)

            if winner == q_player:
                return 1
            elif winner == 0:
                return 0
            return -1

    return 0


def train_episode(
    env: TicTacToeEnv,
    q_agent: QLearningAgent,
    cache: dict[str, int],
    q_player: int,
    config: TrainConfig,
) -> int:
    """Run one training episode with randomly selected opponent."""
    opponent = random.choice(OPPONENT_TYPES)

    if opponent == "self_play":
        return _train_episode_self_play(env, q_agent, q_player, config)

    q_agent.player = q_player
    observation, info = env.reset()
    terminated = False
    last_obs, last_action = None, None

    # Determine how many opponent moves are random
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
                q_agent.learn(last_obs, last_action, 0.0, observation, info, False)

            action = q_agent.act(observation, info, is_training=True)
            last_obs = observation.copy()
            last_action = action
            observation, _, terminated, _, info = env.step(action)

            if terminated:
                winner = env.unwrapped.engine.winner
                reward, result = _get_final_reward(winner, q_player, config)
                q_agent.learn(last_obs, last_action, reward, observation, info, True)
                return result
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
                q_agent.learn(last_obs, last_action, reward, observation, info, True)
                return result

    return 0


# ── Single-Process Training ──────────────────────────────────────────────────


def train_single(
    config: TrainConfig,
    cache: dict[str, int],
    seed: int = 42,
) -> tuple[dict, list[Checkpoint], float]:
    """Returns: (q_table, checkpoints, elapsed_seconds)."""
    random.seed(seed)
    np.random.seed(seed)

    env = TicTacToeEnv()
    q_agent = QLearningAgent(
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon_start,
    )

    checkpoints: list[Checkpoint] = []
    window_w, window_d, window_l = 0, 0, 0

    start = time.perf_counter()

    for episode in tqdm(range(config.total_episodes), desc="  Single", ncols=80):
        q_player = random.choice([1, -1])
        result = train_episode(env, q_agent, cache, q_player, config)

        if result == 1:
            window_w += 1
        elif result == 0:
            window_d += 1
        else:
            window_l += 1

        if (episode + 1) % config.checkpoint_interval == 0:
            total = window_w + window_d + window_l
            checkpoints.append(
                Checkpoint(
                    episode=episode + 1,
                    win_rate=window_w / total * 100,
                    draw_rate=window_d / total * 100,
                    loss_rate=window_l / total * 100,
                    q_table_size=len(q_agent.q_table),
                    epsilon=q_agent.epsilon,
                )
            )
            window_w, window_d, window_l = 0, 0, 0

        if (episode + 1) % config.epsilon_decay_interval == 0:
            q_agent.epsilon = max(
                config.epsilon_end,
                q_agent.epsilon * config.epsilon_decay_rate,
            )

    elapsed = time.perf_counter() - start
    return q_agent.q_table, checkpoints, elapsed


# ── Parallel Training ────────────────────────────────────────────────────────


def _worker_fn(args: tuple) -> tuple[dict, int, int, int]:
    """Worker: train independently, return (q_table, wins, draws, losses)."""
    worker_id, episodes, config_dict, cache, seed = args

    random.seed(seed)
    np.random.seed(seed)

    config = TrainConfig(**config_dict)
    env = TicTacToeEnv()
    q_agent = QLearningAgent(
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon_start,
    )

    wins, draws, losses = 0, 0, 0

    for episode in range(episodes):
        q_player = random.choice([1, -1])
        result = train_episode(env, q_agent, cache, q_player, config)

        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

        if (episode + 1) % config.epsilon_decay_interval == 0:
            q_agent.epsilon = max(
                config.epsilon_end,
                q_agent.epsilon * config.epsilon_decay_rate,
            )

    return q_agent.q_table, wins, draws, losses


def merge_q_tables(tables: list[dict]) -> dict:
    """Merge Q-tables by averaging Q-values across workers."""
    accumulator: dict[str, dict[str, list[float]]] = {}

    for table in tables:
        for state, actions in table.items():
            if state not in accumulator:
                accumulator[state] = {}
            for action, value in actions.items():
                if action not in accumulator[state]:
                    accumulator[state][action] = []
                accumulator[state][action].append(value)

    merged: dict[str, dict[str, float]] = {}
    for state, actions in accumulator.items():
        merged[state] = {
            action: sum(values) / len(values) for action, values in actions.items()
        }
    return merged


def train_parallel(
    config: TrainConfig,
    cache: dict[str, int],
    num_workers: int | None = None,
    seed: int = 123,
) -> tuple[dict, int, int, int, float, int]:
    """Returns: (q_table, wins, draws, losses, elapsed, num_workers)."""
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 16)

    episodes_per_worker = config.total_episodes // num_workers
    remainder = config.total_episodes % num_workers

    config_dict = asdict(config)

    worker_args = []
    for i in range(num_workers):
        eps = episodes_per_worker + (1 if i < remainder else 0)
        worker_args.append((i, eps, config_dict, cache, seed + i))

    print(
        f"  Dispatching {num_workers} workers "
        f"({episodes_per_worker}+ episodes each)..."
    )

    start = time.perf_counter()

    tables: list[dict] = []
    total_w, total_d, total_l = 0, 0, 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_fn, args): args[0] for args in worker_args}
        for future in as_completed(futures):
            q_table, w, d, lo = future.result()
            tables.append(q_table)
            total_w += w
            total_d += d
            total_l += lo
            done_count = len(tables)
            print(
                f"    Worker {done_count}/{num_workers} done " f"(W:{w} D:{d} L:{lo})"
            )

    merged = merge_q_tables(tables)
    elapsed = time.perf_counter() - start

    return merged, total_w, total_d, total_l, elapsed, num_workers


# ── Validation ───────────────────────────────────────────────────────────────


def validate(
    q_table: dict,
    opponent_act_fn,
    q_player: int,
    num_games: int,
) -> tuple[int, int, int]:
    """Play games with no learning/exploration. Returns (wins, draws, losses)."""
    env = TicTacToeEnv()
    q_agent = QLearningAgent(epsilon=0, player=q_player)
    q_agent.q_table = q_table

    wins, draws, losses = 0, 0, 0

    for _ in range(num_games):
        observation, info = env.reset()
        terminated = False

        while not terminated:
            if info["current_player"] == q_player:
                action = q_agent.act(observation, info, is_training=False)
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

    return wins, draws, losses


def run_full_validation(
    q_table: dict,
    cache: dict[str, int],
    label: str,
    games_per_side: int,
) -> dict:
    """Validate against Alpha-Beta, Random, and Self from both sides."""
    random_agent = RandomAgent()
    results = {}

    def ab_fn(obs, _info):
        return cached_ab_act(cache, obs)

    def rand_fn(obs, info):
        return random_agent.act(obs, info)

    for q_player, side_label in [(1, "X"), (-1, "O")]:
        w, d, lo = validate(q_table, ab_fn, q_player, games_per_side)
        key = f"vs_AB_as_{side_label}"
        results[key] = (w, d, lo)
        status = "PASS" if lo == 0 else "FAIL"
        print(
            f"    {label} vs Alpha-Beta (as {side_label}): "
            f"W={w:,} D={d:,} L={lo:,} {status}"
        )

    for q_player, side_label in [(1, "X"), (-1, "O")]:
        w, d, lo = validate(q_table, rand_fn, q_player, games_per_side)
        key = f"vs_Rand_as_{side_label}"
        results[key] = (w, d, lo)
        status = "PASS" if lo == 0 else "FAIL"
        print(
            f"    {label} vs Random    (as {side_label}): "
            f"W={w:,} D={d:,} L={lo:,} {status}"
        )

    for q_player, side_label in [(1, "X"), (-1, "O")]:
        opp = QLearningAgent(epsilon=0, player=-q_player)
        opp.q_table = q_table

        def self_fn(obs, info, _a=opp):
            return _a.act(obs, info, is_training=False)

        w, d, lo = validate(q_table, self_fn, q_player, games_per_side)
        key = f"vs_Self_as_{side_label}"
        results[key] = (w, d, lo)
        status = "PASS" if lo == 0 else "FAIL"
        print(
            f"    {label} vs Self      (as {side_label}): "
            f"W={w:,} D={d:,} L={lo:,} {status}"
        )

    return results


# ── Report ───────────────────────────────────────────────────────────────────


def generate_report(
    config: TrainConfig,
    single_checkpoints: list[Checkpoint],
    single_elapsed: float,
    single_table_size: int,
    single_val: dict,
    parallel_elapsed: float,
    parallel_table_size: int,
    parallel_val: dict,
    num_workers: int,
    parallel_w: int,
    parallel_d: int,
    parallel_l: int,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    speedup = single_elapsed / parallel_elapsed if parallel_elapsed > 0 else 0

    # Sample checkpoints for chart (max ~25 data points)
    step = max(1, len(single_checkpoints) // 25)
    sampled = single_checkpoints[::step]
    if sampled[-1].episode != single_checkpoints[-1].episode:
        sampled.append(single_checkpoints[-1])

    chart_x = json.dumps([cp.episode // 1000 for cp in sampled])
    chart_draw = json.dumps([round(cp.draw_rate, 1) for cp in sampled])
    chart_loss = json.dumps([round(cp.loss_rate, 1) for cp in sampled])

    # Convergence table
    conv_rows = "\n".join(
        f"| {cp.episode:,} | {cp.win_rate:.1f}% | {cp.draw_rate:.1f}% "
        f"| {cp.loss_rate:.1f}% | {cp.q_table_size:,} | {cp.epsilon:.4f} |"
        for cp in single_checkpoints
    )

    # Validation table helper
    def val_rows(val: dict, label: str) -> str:
        lines = []
        for key, (w, d, lo) in val.items():
            opponent, _, _, side = key.split("_")
            opp_name = {"AB": "Alpha-Beta", "Rand": "Random", "Self": "Self"}
            opponent_name = opp_name.get(opponent, opponent)
            status = "PASS" if lo == 0 else "FAIL"
            lines.append(
                f"| {label} vs {opponent_name} (as {side}) "
                f"| {w:,} | {d:,} | {lo:,} | {status} |"
            )
        return "\n".join(lines)

    # Final verdict (must pass ALL opponents)
    all_losses = sum(lo for _, (_, _, lo) in single_val.items())
    verdict = "UNBEATABLE" if all_losses == 0 else "NOT YET CONVERGED"

    return f"""# Q-Learning Training Report

> **Date**: _{today}_
> **Goal**: Train a Tabular Q-Learning agent to achieve **unbeatable play** in Tic-Tac-Toe

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Total Episodes | {config.total_episodes:,} |
| Learning Rate (alpha) | {config.alpha} |
| Discount Factor (gamma) | {config.gamma} |
| Epsilon (start → end) | {config.epsilon_start} → {config.epsilon_end} |
| Epsilon Decay | x{config.epsilon_decay_rate} every {config.epsilon_decay_interval:,} episodes |
| Reward (Win / Draw / Loss) | {config.win_reward} / {config.draw_reward} / {config.loss_reward} |
| Opponent | Mixed: 25% Random / 25% Self-Play / 25% Alpha-Beta / 25% Hybrid (Random 0-{config.max_random_opponent_moves}→AB) |
| Board Normalization | ON (single Q-table handles both X and O) |

---

## How It Learned

### Board Normalization (Canonical Form)

The agent uses **canonical board representation**. When playing as O (player -1), the board is multiplied by -1 before Q-table lookup:

```
Actual board (I am O):      Normalized (Q-table key):
 X | O | .                   O | X | .
 . | X | .         →         . | O | .
 . | . | O                   . | . | X
```

This means **one Q-table serves both perspectives** — training data from playing X and O both contribute to the same Q-values.

### Learning Dynamics

```mermaid
xychart-beta
    title "Training Convergence (per {config.checkpoint_interval:,}-episode window)"
    x-axis "Episodes (x1000)" {chart_x}
    y-axis "Rate (%)" 0 --> 100
    line "Draw Rate" {chart_draw}
    line "Loss Rate" {chart_loss}
```

- **Win Rate > 0%** — when Random prefix creates exploitable positions, agent can win.
- **Convergence = Loss Rate → 0%** — the agent learns to never lose from any position.
- **Early phase** (epsilon ~0.3): High exploration, many losses as agent tries random moves.
- **Mid phase** (epsilon ~0.1): Agent discovers winning/drawing sequences, loss rate drops sharply.
- **Late phase** (epsilon ~0.01): Exploitation mode, agent refines Q-values for known good moves.

### Detailed Convergence Data

| Episode | Win | Draw | Loss | Q-Table Size | Epsilon |
|---------|-----|------|------|--------------|---------|
{conv_rows}

---

## Performance Comparison: Single vs Parallel

| Metric | Single-Process | Parallel ({num_workers} workers) |
|--------|---------------|----------------------------------|
| Wall Time | {single_elapsed:.2f}s | {parallel_elapsed:.2f}s |
| **Speedup** | 1.0x | **{speedup:.1f}x** |
| Q-Table States | {single_table_size:,} | {parallel_table_size:,} |
| Training Result (W/D/L) | _(see convergence)_ | {parallel_w:,} / {parallel_d:,} / {parallel_l:,} |

### Parallel Strategy

- {num_workers} independent workers each train {config.total_episodes // num_workers:,}+ episodes
- Each worker has a different random seed for diverse state exploration
- Q-tables are **merged by averaging** Q-values for shared (state, action) pairs
- Final merged table covers {parallel_table_size:,} unique states

---

## Validation Results ({config.validation_games_per_side:,} games per side)

| Match | Win | Draw | Loss | Status |
|-------|-----|------|------|--------|
{val_rows(single_val, "Single")}
{val_rows(parallel_val, "Parallel")}

---

## Result: **{verdict}**

{'The agent achieves **zero losses** against Alpha-Beta, Random, and itself from both sides. It has learned the complete optimal strategy for Tic-Tac-Toe through pure reinforcement learning.' if all_losses == 0 else 'The agent still has losses. Consider increasing training episodes or tuning hyperparameters.'}
"""


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    project_root = Path(__file__).parent.parent
    cache_path = project_root / "models" / "alphabeta_cache.json"
    model_path = project_root / "models" / "q_table.json"
    report_dir = project_root / ".ai-collab"
    report_dir.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = report_dir / f"training_report_{today}.md"

    config = TrainConfig()

    # Header
    print("=" * 60)
    print(" Q-Learning Training: Unbeatable Tic-Tac-Toe")
    print("=" * 60)
    print(f"  Episodes:  {config.total_episodes:,}")
    print(f"  Alpha:     {config.alpha}  |  Gamma: {config.gamma}")
    print(
        f"  Epsilon:   {config.epsilon_start} → {config.epsilon_end} "
        f"(x{config.epsilon_decay_rate} every {config.epsilon_decay_interval:,})"
    )
    print(
        f"  Reward:    W={config.win_reward} D={config.draw_reward} L={config.loss_reward}"
    )
    print("  Opponent:  Mixed (Random / Self-Play / Alpha-Beta / Hybrid)")
    print(f"  CPU cores: {os.cpu_count()}")
    print()

    # Load cache
    if not cache_path.exists():
        print(f"ERROR: Alpha-Beta cache not found at {cache_path}")
        print("Run `uv run python scripts/precompute_alphabeta.py` first.")
        sys.exit(1)

    cache = load_alphabeta_cache(cache_path)
    print(f"  Loaded Alpha-Beta cache: {len(cache):,} states")
    print()

    # ── Phase 1: Single-Process ──────────────────────────────────────────
    print("-" * 60)
    print(" Phase 1: Single-Process Training")
    print("-" * 60)

    single_table, single_cps, single_time = train_single(config, cache)

    last_cp = single_cps[-1] if single_cps else None
    print(f"\n  Done in {single_time:.2f}s | " f"Q-Table: {len(single_table):,} states")
    if last_cp:
        print(
            f"  Last window: W={last_cp.win_rate:.1f}% "
            f"D={last_cp.draw_rate:.1f}% L={last_cp.loss_rate:.1f}%"
        )
    print()

    # ── Phase 2: Parallel Training ───────────────────────────────────────
    print("-" * 60)
    print(" Phase 2: Parallel Training")
    print("-" * 60)

    (parallel_table, par_w, par_d, par_l, parallel_time, num_workers) = train_parallel(
        config, cache
    )

    print(
        f"\n  Done in {parallel_time:.2f}s | "
        f"Q-Table: {len(parallel_table):,} states (merged)"
    )
    print(f"  Total: W={par_w:,} D={par_d:,} L={par_l:,}")

    speedup = single_time / parallel_time if parallel_time > 0 else 0
    print(
        f"\n  Speedup: {speedup:.1f}x "
        f"(single {single_time:.2f}s → parallel {parallel_time:.2f}s)"
    )
    print()

    # ── Phase 3: Validation ──────────────────────────────────────────────
    print("-" * 60)
    print(f" Phase 3: Validation ({config.validation_games_per_side:,} games/side)")
    print("-" * 60)

    print("  [Single-process model]")
    single_val = run_full_validation(
        single_table,
        cache,
        "Single",
        config.validation_games_per_side,
    )
    print()
    print("  [Parallel model]")
    parallel_val = run_full_validation(
        parallel_table,
        cache,
        "Parallel",
        config.validation_games_per_side,
    )
    print()

    # ── Choose best model (fewer total losses wins) ─────────────────────
    single_total_losses = sum(lo for _, (_, _, lo) in single_val.items())
    parallel_total_losses = sum(lo for _, (_, _, lo) in parallel_val.items())

    if single_total_losses <= parallel_total_losses:
        best_table, best_label = single_table, "single"
    else:
        best_table, best_label = parallel_table, "parallel"

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w") as f:
        json.dump(best_table, f)
    print(f"  Model saved ({best_label}): {model_path}")

    # ── Generate Report ──────────────────────────────────────────────────
    report = generate_report(
        config=config,
        single_checkpoints=single_cps,
        single_elapsed=single_time,
        single_table_size=len(single_table),
        single_val=single_val,
        parallel_elapsed=parallel_time,
        parallel_table_size=len(parallel_table),
        parallel_val=parallel_val,
        num_workers=num_workers,
        parallel_w=par_w,
        parallel_d=par_d,
        parallel_l=par_l,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    # ── Final Verdict ────────────────────────────────────────────────────
    best_losses = min(single_total_losses, parallel_total_losses)
    print()
    print("=" * 60)
    if best_losses == 0:
        print(" RESULT: UNBEATABLE (0 losses vs AB, Random, and Self)")
    else:
        print(f" RESULT: {best_losses} losses remain — needs more training")
    print("=" * 60)


if __name__ == "__main__":
    main()
