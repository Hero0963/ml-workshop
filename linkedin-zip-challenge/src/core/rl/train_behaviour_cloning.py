# src/core/rl/train_behaviour_cloning.py
"""Supervised warm start: train the policy on the solutions the dataset already carries.

Every puzzle in `datasets/rl_datasets_v2` ships with its `solution_path`, and until now that
path has been used for exactly two things -- picking the reverse curriculum's start cell in
`rl_env_v2`, and replaying legal walks in the A0 diagnosis. **It has never been a training
target.** PPO has been rediscovering, from a sparse reward, answers that are already on disk.

That is worth trying because of what the 2026-09-05 measurements say the bottleneck is.
Reading the solve rate as a per-decision accuracy, 6x6 is already right 93.9% of the time and
loses by needing 14.2 correct choices in a row; clearing 0.85 means 98.9%. That is the shape
of a *classification* problem, and there are ~560,000 labelled (state, action) pairs for it
against a 1.17M-parameter network.

Two properties this has and the PPO run does not:

* **Full-length states from the first step.** The reverse curriculum needed 5.5M steps to
  reach k=30 of 36, so states near the true start are the ones it has trained on least. A
  replayed solution visits every position on the path with equal weight.
* **A dense signal.** One label per step instead of one reward per episode.

⚠ **This is a warm start, not a replacement.** Behaviour cloning only ever sees states on the
expert's own trajectory, so the first mistake takes the policy somewhere it has no training
signal for -- the compounding-error problem DAgger exists to solve. Expect it to be strong
where PPO is weak (early-path decisions) and to keep PPO's advantage where it is not.

The model is built by `train_maskable_ppo.build_model`, so what this writes is a drop-in for
`score()`, the baselines and every probe: same architecture, same observation space, same
checkpoint layout.

Run:
    uv run python -m src.core.rl.train_behaviour_cloning --goal goal2_6x6 --run-id bc_6x6
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch as th
from loguru import logger

from src.core.rl.action_space import path_to_actions
from src.core.rl.baselines import make_eval_env
from src.core.rl.rl_env_v2 import DEFAULT_GAMMA, PuzzleSample
from src.core.rl.train_config import GOALS, Goal
from src.core.rl.train_maskable_ppo import (
    CHECKPOINT_DIRNAME,
    FINAL_CHECKPOINT_NAME,
    LOG_ROOT,
    MODEL_ROOT,
    STATE_FILENAME,
    apply_resource_limits,
    build_model,
    describe_splits,
    make_vec_env,
    score,
    select_samples,
)

DEFAULT_SEED = 20260815
DEFAULT_EPOCHS = 8
# Enough val states for the accuracy to be stable between epochs without replaying the
# whole split every time: 300 puzzles is ~10,500 decisions on 6x6.
VAL_PUZZLES_PER_EPOCH = 300
# Below this the mask leaves no decision to make, so the state says nothing about the policy.
MIN_LEGAL_ACTIONS_FOR_A_CHOICE = 2
PROGRESS_FILENAME = "bc_progress.jsonl"
# Matches PPO's own `vf_coef`, so the critic is weighted the same way here as in the
# fine-tune that consumes these weights.
DEFAULT_VALUE_COEF = 0.5


class SupervisedPair:
    """One labelled decision, plus the return the expert collected from here on.

    `value_target` exists so the value head is not left random. PPO estimates advantages
    from V(s); starting a fine-tune with an untrained value head means the first updates
    are driven by noise, which is how a good cloned policy gets destroyed before the
    critic catches up.
    """

    __slots__ = ("observation", "action_mask", "action", "value_target")

    def __init__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action: int,
        value_target: float = 0.0,
    ) -> None:
        self.observation = observation
        self.action_mask = action_mask
        self.action = action
        self.value_target = value_target


def iter_supervised_pairs(
    samples: Sequence[PuzzleSample],
    connectivity_features: bool,
    shaping_lambda: float = 0.0,
    gamma: float = DEFAULT_GAMMA,
) -> Iterator[SupervisedPair]:
    """Replays each solution through the env, yielding every decision along the way.

    The env is built through `baselines.make_eval_env`, which is the project's single
    evaluation construction point, so the observations here cannot drift from the ones the
    policy is scored on -- the failure that cost three experiment arms in the P0 run.

    `value_target` is the *observed* discounted return of the replay, accumulated backwards
    once the episode is done, rather than an analytic `gamma ** remaining`: the shaping term
    is part of the reward the fine-tuning env pays out, and re-deriving it here would be a
    second copy of the env's reward rule.

    ⚠ This is V under the *expert*, not under our policy. The expert always succeeds, so
    the target is optimistic -- but its ordering across states is right, which is what
    advantage estimation mostly needs, and PPO regresses the bias away quickly.
    """
    for sample in samples:
        env = make_eval_env(
            sample,
            connectivity_features=connectivity_features,
            shaping_lambda=shaping_lambda,
            gamma=gamma,
        )
        env.reset(seed=DEFAULT_SEED)
        trajectory: list[SupervisedPair] = []
        rewards: list[float] = []
        for action in path_to_actions(sample.solution_path):
            mask = env.action_masks()
            trajectory.append(SupervisedPair(env.observation(), mask.copy(), action))
            _, reward, _, _, _ = env.step(action)
            rewards.append(float(reward))

        discounted = 0.0
        for pair, reward in zip(reversed(trajectory), reversed(rewards)):
            discounted = reward + gamma * discounted
            pair.value_target = discounted
        yield from trajectory


def _stack(
    pairs: Sequence[SupervisedPair],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    observation = {
        key: np.stack([pair.observation[key] for pair in pairs])
        for key in pairs[0].observation
    }
    masks = np.stack([pair.action_mask for pair in pairs])
    actions = np.array([pair.action for pair in pairs], dtype=np.int64)
    returns = np.array([pair.value_target for pair in pairs], dtype=np.float32)
    return observation, masks, actions, returns


def _batches(
    pairs: Iterator[SupervisedPair], batch_size: int
) -> Iterator[list[SupervisedPair]]:
    batch: list[SupervisedPair] = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def evaluate_action_accuracy(
    model, samples: Sequence[PuzzleSample], goal: Goal, batch_size: int
) -> dict[str, float]:
    """Agreement with the solution, reported over all decisions and over real choices.

    **Only `choice_accuracy` is comparable to anything.** 69.1% of 4x4 decisions and 59.4%
    of 6x6 ones have a single legal move, and the mask makes those free, so an accuracy over
    all decisions is mostly a measure of how forced the board is: an untrained policy already
    scores ~0.69 on 4x4. Restricting to states with two or more legal moves is the quantity
    the solve rate is an exponential of.

    ⚠ Both are measured on *expert* states -- the ones the solution walks through. A policy
    that leaves the expert's path lands in states this number says nothing about, which is
    exactly why behaviour cloning's held-out accuracy overstates its solve rate.
    """
    correct = total = 0
    choice_correct = choice_total = 0
    pairs = iter_supervised_pairs(samples, goal.connectivity_features)
    for batch in _batches(pairs, batch_size):
        observation, masks, actions, _ = _stack(batch)
        with th.no_grad():
            obs_tensor, _ = model.policy.obs_to_tensor(observation)
            distribution = model.policy.get_distribution(obs_tensor, action_masks=masks)
            predicted = distribution.distribution.probs.argmax(dim=1).cpu().numpy()
        hit = predicted == actions
        correct += int(hit.sum())
        total += len(actions)
        is_choice = masks.sum(axis=1) >= MIN_LEGAL_ACTIONS_FOR_A_CHOICE
        choice_correct += int(hit[is_choice].sum())
        choice_total += int(is_choice.sum())
    return {
        "action_accuracy": correct / total if total else 0.0,
        "choice_accuracy": choice_correct / choice_total if choice_total else 0.0,
        "choice_share": choice_total / total if total else 0.0,
    }


def train_epoch(
    model, pairs: Iterator[SupervisedPair], batch_size: int, value_coef: float
) -> tuple[dict[str, float], int]:
    """One pass of masked cross-entropy over the replay, plus value regression.

    The value term is what makes this a *warm start* rather than just a good policy: PPO
    derives its advantages from V(s), so handing the fine-tune an untrained critic means
    the first updates are noise -- and the cloned policy is what that noise destroys.
    """
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    steps = 0
    for batch in _batches(pairs, batch_size):
        observation, masks, actions, returns = _stack(batch)
        obs_tensor, _ = model.policy.obs_to_tensor(observation)
        device = obs_tensor["grid"].device
        distribution = model.policy.get_distribution(obs_tensor, action_masks=masks)
        # The masked log-prob is exactly what inference computes, so the loss optimises the
        # quantity that is actually used rather than an unmasked proxy.
        policy_loss = -distribution.log_prob(
            th.as_tensor(actions, device=device)
        ).mean()
        values = model.policy.predict_values(obs_tensor).flatten()
        value_loss = th.nn.functional.mse_loss(
            values, th.as_tensor(returns, device=device)
        )
        loss = policy_loss + value_coef * value_loss

        model.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
        model.policy.optimizer.step()

        totals["loss"] += float(loss.item())
        totals["policy_loss"] += float(policy_loss.item())
        totals["value_loss"] += float(value_loss.item())
        steps += 1
    divisor = steps or 1
    return {key: value / divisor for key, value in totals.items()}, steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True, choices=sorted(GOALS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--value-coef",
        type=float,
        default=DEFAULT_VALUE_COEF,
        help="Weight on the value regression; 0 leaves the critic untrained.",
    )
    parser.add_argument(
        "--train-puzzles",
        type=int,
        default=None,
        help="Cap the training split, for a quick pilot.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    goal = GOALS[args.goal]
    limits = apply_resource_limits(goal.resources)
    th.manual_seed(args.seed)
    random.seed(args.seed)

    run_dir = LOG_ROOT / args.run_id
    checkpoint_dir = MODEL_ROOT / args.run_id / CHECKPOINT_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    split_counts = describe_splits(goal)
    train_samples = select_samples(goal, "train")
    if args.train_puzzles is not None:
        train_samples = train_samples[: args.train_puzzles]
    val_samples = select_samples(goal, "val")[:VAL_PUZZLES_PER_EPOCH]

    # The vec env is only here because `build_model` needs the spaces; behaviour cloning
    # never steps it. Building the model this way is what keeps the result a drop-in.
    vec_env = make_vec_env(train_samples, goal, args.seed, None, goal.resources.vec_env)
    model = build_model(goal, vec_env, run_dir, args.seed, args.device)
    logger.info(
        f"Behaviour cloning {args.goal} on {len(train_samples)} puzzles, "
        f"{args.epochs} epochs, batch {goal.ppo.batch_size}"
    )

    progress_path = run_dir / PROGRESS_FILENAME
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        order = list(train_samples)
        random.shuffle(order)
        epoch_started = time.perf_counter()
        losses, steps = train_epoch(
            model,
            iter_supervised_pairs(
                order,
                goal.connectivity_features,
                shaping_lambda=goal.shaping_lambda,
                gamma=goal.ppo.gamma,
            ),
            goal.ppo.batch_size,
            args.value_coef,
        )
        accuracy = evaluate_action_accuracy(
            model, val_samples, goal, goal.ppo.batch_size
        )
        row = {
            "epoch": epoch,
            **{key: round(value, 6) for key, value in losses.items()},
            "gradient_steps": steps,
            **{f"val_{key}": round(value, 6) for key, value in accuracy.items()},
            "seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.append(row)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        logger.info(
            f"epoch {epoch}/{args.epochs} policy_loss={losses['policy_loss']:.4f} "
            f"value_loss={losses['value_loss']:.4f} "
            f"val_choice_accuracy={accuracy['choice_accuracy']:.4f} "
            f"(all decisions {accuracy['action_accuracy']:.4f}) ({row['seconds']}s)"
        )

    training_seconds = round(time.perf_counter() - started, 1)
    model.save(checkpoint_dir / FINAL_CHECKPOINT_NAME)
    logger.success(f"Saved {checkpoint_dir / FINAL_CHECKPOINT_NAME}.zip")

    eval_samples = select_samples(goal, args.eval_split)
    scores = score(
        model,
        eval_samples,
        seed=args.seed,
        episodes_per_puzzle=args.eval_episodes,
        with_baselines=True,
        connectivity_features=goal.connectivity_features,
    )
    report = {
        "run_id": args.run_id,
        "method": "behaviour_cloning",
        "config": {
            "goal": {
                "key": goal.key,
                "sizes": list(goal.sizes),
                "dataset": goal.dataset,
                "target_solve_rate": goal.target_solve_rate,
                "connectivity_features": goal.connectivity_features,
            },
            "epochs": args.epochs,
            "value_coef": args.value_coef,
            "batch_size": goal.ppo.batch_size,
            "learning_rate": goal.ppo.learning_rate,
            "train_puzzles": len(train_samples),
            "split_counts": split_counts,
            "seed": args.seed,
            "resource_limits": limits,
        },
        "training_seconds": training_seconds,
        "history": history,
        "eval_split": args.eval_split,
        "eval_puzzles": len(eval_samples),
        "target_solve_rate": goal.target_solve_rate,
        "scores": scores,
        "meets_target": bool(
            scores["model"]["overall"]["solve_rate"] >= goal.target_solve_rate
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / f"eval_{args.eval_split}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    Path(run_dir / STATE_FILENAME).write_text(
        json.dumps({"config": report["config"], "history": history}, indent=2),
        encoding="utf-8",
    )
    overall = scores["model"]["overall"]
    logger.success(
        f"{args.run_id}: solve={overall['solve_rate']:.4f} "
        f"dead_end={overall['dead_end_rate']:.4f} "
        f"greedy={scores['greedy']['overall']['solve_rate']:.4f} "
        f"n={overall['episodes']} in {training_seconds}s"
    )


if __name__ == "__main__":
    main()
