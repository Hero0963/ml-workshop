# src/core/rl/collect_solutions.py
"""Expert Iteration's collection step: let the policy search, keep only what the checker accepts.

Behaviour cloning has one label per puzzle -- the path the generator built it from -- but
these puzzles often have more than one solution, and the cloned policy already finds some
of the others: on 4x4 it agrees with the label on 88.11% of choices yet solves 89.47% of
the puzzles. Where a puzzle has several solutions, training on one of them teaches the
policy that every valid alternative at a fork is *wrong*, which is exactly the pressure
that makes a longer-trained policy sharper and costs it best-of-N (2026-09-12: 6 epochs
beat 10 on 6x6 best-of-32, 0.9465 vs 0.8535).

So the search that is already paid for at inference (best-of-N sampling) is run over the
*training* split, every walk the env scores as solved is re-checked by the project's
independent scorer (`calculate_fitness_score`, the one the heuristic solvers use), and
the distinct solutions that differ from the dataset's are written out. Behaviour cloning
then picks, per puzzle and per epoch, one of the known solutions uniformly
(`train_behaviour_cloning --extra-solutions`). That is the "apprentice imitates the
search" half of Expert Iteration (Anthony et al., 2017); LLM work calls the same loop
rejection-sampling fine-tuning.

Sampling is batched -- hundreds of episodes in flight, one forward pass per step for all
of them -- because the probes' one-observation-at-a-time loop runs ~20 episodes a second,
and the training split is 47,435 puzzles. Actions are drawn from the same masked
distribution `model.predict(deterministic=False)` samples from.

Run:
    uv run python -m src.core.rl.collect_solutions --run-id bc_multi_456_e6 \
        --goal goal3_multi --attempts 16 --collection-id exit_r1
"""

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch as th
from loguru import logger
from sb3_contrib import MaskablePPO

from src.core.rl.action_space import ACTION_DELTAS
from src.core.rl.baselines import make_eval_env
from src.core.rl.generate_dataset_v2 import sample_fingerprint
from src.core.rl.rl_env_v2 import NUM_ACTIONS, PuzzleEnvV2, PuzzleSample
from src.core.rl.train_config import GOALS
from src.core.rl.train_maskable_ppo import (
    CHECKPOINT_DIRNAME,
    FINAL_CHECKPOINT_NAME,
    MODEL_ROOT,
    apply_resource_limits,
    select_samples,
)
from src.core.utils import calculate_fitness_score

COLLECTION_ROOT = Path(__file__).resolve().parents[3] / "logs" / "rl_exit"
SOLUTIONS_FILENAME = "solutions.json"
SUMMARY_FILENAME = "summary.json"
DEFAULT_SEED = 20260815
DEFAULT_ATTEMPTS = 16
DEFAULT_PARALLEL_EPISODES = 1024
PROGRESS_EVERY_EPISODES = 50_000

Cell = tuple[int, int]
Walk = tuple[Cell, ...]


def sample_key(sample: PuzzleSample) -> str:
    """Stable id for a dataset sample, from the same canonical text the digests use."""
    return hashlib.sha256(sample_fingerprint(sample).encode("utf-8")).hexdigest()


def is_valid_solution(sample: PuzzleSample, walk: Sequence[Cell]) -> bool:
    """A second opinion that shares no code with the env's own `_is_solved`."""
    score, perfect = calculate_fitness_score(sample.puzzle, list(walk))
    return score == perfect


def sample_actions(
    probs: np.ndarray, masks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Inverse-CDF draw per row, restricted to legal actions.

    The masked distribution already puts ~0 on illegal actions; zeroing them exactly and
    pinning the CDF to 1 from the last legal action onward means float rounding can never
    hand back an illegal move. Draws are in (0, 1] so a zero-probability first action is
    never chosen by a draw of exactly 0.
    """
    legal = np.where(masks, probs, 0.0)
    legal = legal / legal.sum(axis=1, keepdims=True)
    cumulative = legal.cumsum(axis=1)
    last_legal = NUM_ACTIONS - 1 - np.argmax(masks[:, ::-1], axis=1)
    cumulative[np.arange(NUM_ACTIONS)[None, :] >= last_legal[:, None]] = 1.0
    draws = 1.0 - rng.random(len(probs))
    return np.argmax(cumulative >= draws[:, None], axis=1)


@dataclass
class PuzzleOutcome:
    attempts: int = 0
    solved_attempts: int = 0
    walks: set[Walk] = field(default_factory=set)
    rejected_by_checker: int = 0


@dataclass
class _Episode:
    index: int
    env: PuzzleEnvV2
    walk: list[Cell]
    observation: dict[str, np.ndarray]


def sample_policy_walks(
    model: MaskablePPO,
    samples: Sequence[PuzzleSample],
    attempts: int,
    rng: np.random.Generator,
    parallel: int = DEFAULT_PARALLEL_EPISODES,
) -> list[PuzzleOutcome]:
    """`attempts` sampled episodes per puzzle; returns the distinct solving walks of each.

    Every attempt runs to the end (no stopping at the first solve): the point is to find
    *different* solutions, not to measure best-of-N.
    """
    outcomes = [PuzzleOutcome() for _ in samples]
    jobs = [index for index in range(len(samples)) for _ in range(attempts)]
    next_job = 0
    active: list[_Episode] = []
    finished = 0
    started = time.perf_counter()

    while next_job < len(jobs) or active:
        while len(active) < parallel and next_job < len(jobs):
            index = jobs[next_job]
            next_job += 1
            outcomes[index].attempts += 1
            env = make_eval_env(samples[index])
            observation, _ = env.reset(seed=DEFAULT_SEED)
            if not env.action_masks().any():
                finished += 1
                continue
            start = tuple(samples[index].solution_path[0])
            active.append(_Episode(index, env, [start], observation))

        if not active:
            break
        observation = {
            key: np.stack([episode.observation[key] for episode in active])
            for key in active[0].observation
        }
        masks = np.stack([episode.env.action_masks() for episode in active])
        with th.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(observation)
            distribution = model.policy.get_distribution(obs_tensor, action_masks=masks)
            probs = distribution.distribution.probs.cpu().numpy().astype(np.float64)
        actions = sample_actions(probs, masks, rng)

        still_running: list[_Episode] = []
        for episode, action in zip(active, actions):
            episode.observation, _, terminated, truncated, info = episode.env.step(
                int(action)
            )
            row, col = episode.walk[-1]
            delta_row, delta_col = ACTION_DELTAS[int(action)]
            episode.walk.append((row + delta_row, col + delta_col))
            if not (terminated or truncated):
                still_running.append(episode)
                continue
            finished += 1
            if finished % PROGRESS_EVERY_EPISODES == 0:
                logger.info(
                    f"{finished}/{len(jobs)} episodes "
                    f"({time.perf_counter() - started:.0f}s)"
                )
            if not info["solved"]:
                continue
            outcome = outcomes[episode.index]
            outcome.solved_attempts += 1
            walk = tuple(episode.walk)
            if walk in outcome.walks:
                continue
            if is_valid_solution(samples[episode.index], walk):
                outcome.walks.add(walk)
            else:
                outcome.rejected_by_checker += 1
        active = still_running
    return outcomes


def alternative_walks(sample: PuzzleSample, outcome: PuzzleOutcome) -> list[Walk]:
    """Solutions the dataset does not already carry, in a stable order."""
    dataset_walk = tuple(tuple(cell) for cell in sample.solution_path)
    return sorted(walk for walk in outcome.walks if walk != dataset_walk)


def summarise(
    samples: Sequence[PuzzleSample], outcomes: Sequence[PuzzleOutcome]
) -> dict[str, Any]:
    by_size: dict[int, list[tuple[PuzzleSample, PuzzleOutcome]]] = defaultdict(list)
    for sample, outcome in zip(samples, outcomes):
        by_size[sample.puzzle["grid_size"][0]].append((sample, outcome))

    def stats(rows: list[tuple[PuzzleSample, PuzzleOutcome]]) -> dict[str, Any]:
        alternatives = [len(alternative_walks(s, o)) for s, o in rows]
        solved_any = [o for _, o in rows if o.solved_attempts]
        dataset_walk_found = sum(
            1
            for s, o in rows
            if tuple(tuple(cell) for cell in s.solution_path) in o.walks
        )
        return {
            "puzzles": len(rows),
            "solved_by_any_attempt": len(solved_any) / len(rows),
            "sample_solve_rate": sum(o.solved_attempts for _, o in rows)
            / sum(o.attempts for _, o in rows),
            "puzzles_with_an_alternative": sum(1 for a in alternatives if a)
            / len(rows),
            "puzzles_with_an_alternative_when_solved": (
                sum(
                    1
                    for (s, o), a in zip(rows, alternatives)
                    if o.solved_attempts and a
                )
                / len(solved_any)
                if solved_any
                else None
            ),
            "mean_distinct_solutions_when_solved": (
                statistics.fmean(len(o.walks) for o in solved_any)
                if solved_any
                else None
            ),
            "dataset_solution_among_walks_when_solved": (
                dataset_walk_found / len(solved_any) if solved_any else None
            ),
            "alternatives_total": sum(alternatives),
            "alternatives_histogram": {
                str(k): alternatives.count(k) for k in sorted(set(alternatives))
            },
            "rejected_by_checker": sum(o.rejected_by_checker for _, o in rows),
        }

    return {
        "overall": stats([row for rows in by_size.values() for row in rows]),
        "per_size": {size: stats(rows) for size, rows in sorted(by_size.items())},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Collector policy's run id.")
    parser.add_argument("--checkpoint", default=FINAL_CHECKPOINT_NAME)
    parser.add_argument("--goal", required=True, choices=sorted(GOALS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--size", type=int, default=None, help="Keep one board size.")
    parser.add_argument("--limit", type=int, default=None, help="First N puzzles only.")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--collection-id", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    goal = GOALS[args.goal]
    limits = apply_resource_limits(goal.resources)
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    samples = select_samples(goal, args.split)
    if args.size is not None:
        samples = [s for s in samples if s.puzzle["grid_size"][0] == args.size]
    if args.limit is not None:
        samples = samples[: args.limit]

    checkpoint = (
        MODEL_ROOT / args.run_id / CHECKPOINT_DIRNAME / f"{args.checkpoint}.zip"
    )
    model = MaskablePPO.load(checkpoint, device=args.device)
    logger.info(
        f"Collecting {args.attempts} attempts x {len(samples)} puzzles "
        f"with {args.run_id}/{args.checkpoint}"
    )

    started = time.perf_counter()
    outcomes = sample_policy_walks(model, samples, args.attempts, rng, args.parallel)
    seconds = round(time.perf_counter() - started, 1)

    solutions = {
        sample_key(sample): [[list(cell) for cell in walk] for walk in alternatives]
        for sample, outcome in zip(samples, outcomes)
        if (alternatives := alternative_walks(sample, outcome))
    }
    summary = {
        "collection_id": args.collection_id,
        "collector": {"run_id": args.run_id, "checkpoint": args.checkpoint},
        "goal": args.goal,
        "dataset": goal.dataset,
        "split": args.split,
        "size": args.size,
        "limit": args.limit,
        "attempts": args.attempts,
        "seed": args.seed,
        "resource_limits": limits,
        "seconds": seconds,
        "stats": summarise(samples, outcomes),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = COLLECTION_ROOT / args.collection_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SOLUTIONS_FILENAME).write_text(json.dumps(solutions), encoding="utf-8")
    (out_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    overall = summary["stats"]["overall"]
    logger.success(
        f"{args.collection_id}: {overall['puzzles']} puzzles, "
        f"solved by any attempt {overall['solved_by_any_attempt']:.4f}, "
        f"with an alternative {overall['puzzles_with_an_alternative']:.4f}, "
        f"{overall['alternatives_total']} alternatives, "
        f"rejected by checker {overall['rejected_by_checker']} in {seconds}s"
    )


if __name__ == "__main__":
    main()
