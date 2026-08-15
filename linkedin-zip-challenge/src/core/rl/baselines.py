# src/core/rl/baselines.py
"""The two control policies A2 has to beat, per the track plan §6.

Run:  uv run python -m src.core.rl.baselines --dataset <dir under datasets/rl_datasets_v2>

* **masked random** -- uniform over the legal actions. With masking this can be
  surprisingly strong on small boards, which is exactly why it must be measured before
  any training claim is made.
* **greedy** -- always move closer to the next number, tie-broken at random. This is the
  ceiling of what distance-based reward shaping could ever teach, i.e. the strategy the
  2025-10 run was accidentally optimising for.

Both are evaluated under the same one-stroke rules as the agent, so their solve rate is
directly comparable to `deterministic=True` evaluation of a trained policy.
"""

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from loguru import logger

from src.core.rl.action_space import ACTION_DELTAS
from src.core.rl.rl_env_v2 import PuzzleEnvV2, PuzzleSample

DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "rl_datasets_v2"
ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "logs" / "rl_baselines"
DEFAULT_SEED = 20260815

PolicyFn = Callable[[PuzzleEnvV2, np.random.Generator], int]


def masked_random_action(env: PuzzleEnvV2, rng: np.random.Generator) -> int:
    """Uniform over the legal actions."""
    legal = np.flatnonzero(env.action_masks())
    return int(rng.choice(legal))


def greedy_action(env: PuzzleEnvV2, rng: np.random.Generator) -> int:
    """Legal move that minimises the Manhattan distance to the next number."""
    legal = np.flatnonzero(env.action_masks())
    target = env.num_map.get(env._next_waypoint)
    if target is None:
        return int(rng.choice(legal))

    row, col = env._agent_location
    distances = []
    for action in legal:
        delta_row, delta_col = ACTION_DELTAS[int(action)]
        distances.append(
            abs(row + delta_row - target[0]) + abs(col + delta_col - target[1])
        )
    best = legal[np.flatnonzero(np.asarray(distances) == min(distances))]
    return int(rng.choice(best))


POLICIES: dict[str, PolicyFn] = {
    "masked_random": masked_random_action,
    "greedy": greedy_action,
}


def run_episode(
    env: PuzzleEnvV2, policy: PolicyFn, rng: np.random.Generator
) -> dict[str, Any]:
    """Plays one episode and reports how it ended."""
    _, info = env.reset(seed=int(rng.integers(2**31 - 1)))
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        if not env.action_masks().any():
            break
        action = policy(env, rng)
        _, _, terminated, truncated, info = env.step(action)
        steps += 1

    solved = bool(info["solved"])
    return {
        "solved": solved,
        # An episode ends solved, out of budget, or stuck: everything else is a dead end.
        "dead_end": not solved and not truncated,
        "truncated": bool(truncated),
        "coverage": float(info["coverage"]),
        "steps": steps,
    }


def evaluate(
    samples: Sequence[PuzzleSample],
    policy_name: str,
    seed: int = DEFAULT_SEED,
    episodes_per_puzzle: int = 1,
    reverse_curriculum_k: int | None = None,
) -> dict[str, Any]:
    """Runs one policy over every sample and aggregates overall and per-size."""
    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed)
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        env = PuzzleEnvV2(
            [sample], reverse_curriculum_k=reverse_curriculum_k, shaping_lambda=0.0
        )
        size = sample.puzzle["grid_size"][0]
        for _ in range(episodes_per_puzzle):
            by_size[size].append(run_episode(env, policy, rng))

    def summarise(episodes: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(episodes)
        return {
            "episodes": count,
            "solve_rate": sum(e["solved"] for e in episodes) / count,
            "dead_end_rate": sum(e["dead_end"] for e in episodes) / count,
            "truncation_rate": sum(e["truncated"] for e in episodes) / count,
            "mean_coverage": sum(e["coverage"] for e in episodes) / count,
            "mean_steps": sum(e["steps"] for e in episodes) / count,
        }

    all_episodes = [episode for episodes in by_size.values() for episode in episodes]
    return {
        "policy": policy_name,
        "seed": seed,
        "reverse_curriculum_k": reverse_curriculum_k,
        "overall": summarise(all_episodes),
        "per_size": {
            size: summarise(episodes) for size, episodes in sorted(by_size.items())
        },
    }


def load_split(dataset_dir: Path, split: str) -> list[PuzzleSample]:
    with (dataset_dir / "dataset.pkl").open("rb") as handle:
        dataset = pickle.load(handle)
    return dataset["splits"][split]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes-per-puzzle", type=int, default=1)
    parser.add_argument(
        "--reverse-curriculum-k",
        type=int,
        default=None,
        help="Start k cells from the end of the solution; omit for the true start.",
    )
    args = parser.parse_args()

    dataset_dir = DATASET_ROOT / args.dataset
    samples = load_split(dataset_dir, args.split)
    logger.info(f"Evaluating {len(samples)} puzzles from {dataset_dir} [{args.split}]")

    results = {
        name: evaluate(
            samples,
            name,
            seed=args.seed,
            episodes_per_puzzle=args.episodes_per_puzzle,
            reverse_curriculum_k=args.reverse_curriculum_k,
        )
        for name in POLICIES
    }

    for name, result in results.items():
        overall = result["overall"]
        logger.info(
            f"{name:14s} solve={overall['solve_rate']:.3f} "
            f"dead_end={overall['dead_end_rate']:.3f} "
            f"coverage={overall['mean_coverage']:.3f} "
            f"episodes={overall['episodes']}"
        )
        for size, stats in result["per_size"].items():
            logger.info(
                f"   size {size}: solve={stats['solve_rate']:.3f} "
                f"dead_end={stats['dead_end_rate']:.3f} "
                f"coverage={stats['mean_coverage']:.3f} (n={stats['episodes']})"
            )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"baselines_{args.dataset}_{args.split}.json"
    artifact.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(dataset_dir),
                "split": args.split,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.success(f"Baseline results written to {artifact}")


if __name__ == "__main__":
    main()
