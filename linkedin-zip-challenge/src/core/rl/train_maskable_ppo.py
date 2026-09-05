# src/core/rl/train_maskable_ppo.py
"""MaskablePPO training for the one-stroke Zip environment (roadmap item 3, stage A2).

Run:
    uv run python -m src.core.rl.train_maskable_ppo --goal goal1_4x4
    uv run python -m src.core.rl.train_maskable_ppo --goal goal2_6x6 --resume

Board size, wall policy, hyperparameters and the done condition all live in
`src.core.rl.train_config`; this module only executes a goal. Command line flags cover the
things that change between runs of the *same* goal (how long, which device, resume).

Three things this module exists to get right, all of them lessons the track already paid
for:

*   **The grid must reach a CNN.** SB3 chooses between `NatureCNN` and a plain flatten by
    calling `is_image_space`, which demands `uint8` in 0-255; our stack is `float32` in
    0-1, so the default `MultiInputPolicy` silently **flattens** the 8x8x8 observation and
    throws the board geometry away -- it does not fail, it just learns from a 512-vector.
    Forcing the image path instead (`normalized_image=True`) crashes, because `NatureCNN`
    opens with an 8x8 stride-4 convolution and an 8x8 board leaves a 1x1 feature map for
    the next 4x4 kernel. Both measured 2026-08-29. Hence `GridScalarExtractor`.
*   **The curriculum must be resumable.** An SB3 checkpoint stores policy and optimiser
    state but knows nothing about `reverse_curriculum_k`, so resuming from the model alone
    silently restarts the curriculum at `k_start` while the loss curve still looks healthy.
    `train_state.json` is written beside every checkpoint and carries k, the success
    window and the promotion history.
*   **Everything measurable is written down.** `progress.jsonl` is the human-readable
    record (one row per rollout: k, success rate, dead-end rate, fps, VRAM), tensorboard
    holds the curves, and `episodes.jsonl` (opt-in) holds per-episode outcomes for
    attribution. Checkpoints are never pruned: the VLM track lost a whole experiment to
    `save_total_limit=2`.
"""

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch as th
from loguru import logger
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch import nn

from src.core.rl.baselines import (
    DATASET_ROOT,
    DEFAULT_SEED,
    POLICIES,
    evaluate,
    load_split,
)
from src.core.rl.rl_env_v2 import PuzzleEnvV2, PuzzleSample
from src.core.rl.train_config import (
    GOALS,
    VEC_ENV_KINDS,
    Goal,
    NetworkSettings,
    ResourceSettings,
)

LOG_ROOT = Path(__file__).resolve().parents[3] / "logs" / "rl_a2"
MODEL_ROOT = Path(__file__).resolve().parents[3] / "models" / "rl_a2"

STATE_FILENAME = "train_state.json"
PROGRESS_FILENAME = "progress.jsonl"
EPISODE_FILENAME = "episodes.jsonl"
CHECKPOINT_DIRNAME = "checkpoints"
TENSORBOARD_DIRNAME = "tb"
FINAL_CHECKPOINT_NAME = "model_final"
BYTES_PER_MIB = 1024 * 1024
NO_CURRICULUM_TB_VALUE = -1

WALL_FILTERS: dict[str, Callable[[PuzzleSample], bool]] = {
    "all": lambda sample: True,
    "none": lambda sample: not sample.puzzle["walls"],
    "only": lambda sample: bool(sample.puzzle["walls"]),
}


class GridScalarExtractor(BaseFeaturesExtractor):
    """Padded 3x3 convolutions over the board, concatenated with the scalar vector.

    No pooling: the board is 8x8 and every cell position matters, so downsampling would
    discard exactly the information the policy needs (restart plan section 4.7).
    """

    def __init__(
        self,
        observation_space: Any,
        network: NetworkSettings | None = None,
    ):
        network = network or NetworkSettings()
        super().__init__(observation_space, network.features_dim)
        channels, height, width = observation_space["grid"].shape
        convolutions: list[nn.Module] = []
        in_channels = channels
        for _ in range(3):
            convolutions.append(
                nn.Conv2d(
                    in_channels,
                    network.conv_channels,
                    network.conv_kernel,
                    padding=network.conv_padding,
                )
            )
            convolutions.append(nn.ReLU())
            in_channels = network.conv_channels
        convolutions.append(nn.Flatten())
        self.cnn = nn.Sequential(*convolutions)

        scalars = observation_space["scalars"].shape[0]
        flattened = network.conv_channels * height * width + scalars
        self.linear = nn.Sequential(
            nn.Linear(flattened, network.features_dim), nn.ReLU()
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        grid = self.cnn(observations["grid"])
        return self.linear(th.cat([grid, observations["scalars"]], dim=1))


@dataclass
class EpisodeOutcome:
    """How one episode ended, in the three buckets `baselines.run_episode` also uses."""

    solved: bool
    dead_end: bool
    truncated: bool
    steps: int
    coverage: float


def classify_episode(info: dict[str, Any]) -> EpisodeOutcome:
    solved = bool(info.get("solved", False))
    dead_end = bool(info.get("dead_end", False))
    return EpisodeOutcome(
        solved=solved,
        dead_end=dead_end,
        truncated=not solved and not dead_end,
        steps=int(info.get("steps", 0)),
        coverage=float(info.get("coverage", 0.0)),
    )


@dataclass
class CurriculumState:
    """Everything a resume needs that the SB3 checkpoint does not carry."""

    current_k: int | None
    window: list[bool] = field(default_factory=list)
    promotions: list[dict[str, Any]] = field(default_factory=list)
    episodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_k": self.current_k,
            "window": self.window,
            "promotions": self.promotions,
            "episodes": self.episodes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CurriculumState":
        return cls(
            current_k=payload["current_k"],
            window=list(payload.get("window", [])),
            promotions=list(payload.get("promotions", [])),
            episodes=int(payload.get("episodes", 0)),
        )


def next_curriculum_k(current_k: int, k_step: int, max_path_length: int) -> int | None:
    """The next start distance, or `None` once the agent should start from the real start.

    `PuzzleEnvV2` clamps a `k` longer than the solution to the true start anyway, so
    collapsing it to `None` here keeps the recorded curriculum honest about being finished.
    """
    next_k = current_k + k_step
    return None if next_k >= max_path_length else next_k


class CurriculumCallback(BaseCallback):
    """Advances the reverse curriculum on measured success and records every rollout."""

    def __init__(
        self,
        state: CurriculumState,
        goal: Goal,
        max_path_length: int,
        progress_path: Path,
        episode_path: Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.state = state
        self.settings = goal.curriculum
        self.max_path_length = max_path_length
        self.window: deque[bool] = deque(
            state.window, maxlen=self.settings.promote_window
        )
        self.progress_path = progress_path
        self.episode_path = episode_path
        self._recent: list[EpisodeOutcome] = []
        self._started_at = time.perf_counter()
        self._timesteps_at_start = 0

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()
        self._timesteps_at_start = self.model.num_timesteps
        self.training_env.env_method("set_reverse_curriculum_k", self.state.current_k)
        logger.info(f"Curriculum starts at k={self.state.current_k}")

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            outcome = classify_episode(info)
            self.window.append(outcome.solved)
            self._recent.append(outcome)
            self.state.episodes += 1
            if self.episode_path is not None:
                self._append_jsonl(
                    self.episode_path,
                    {
                        "timesteps": self.model.num_timesteps,
                        "k": self.state.current_k,
                        "solved": outcome.solved,
                        "dead_end": outcome.dead_end,
                        "truncated": outcome.truncated,
                        "steps": outcome.steps,
                        "coverage": round(outcome.coverage, 4),
                    },
                )
        self._maybe_promote()
        return True

    def _maybe_promote(self) -> None:
        if self.state.current_k is None or len(self.window) < self.window.maxlen:
            return
        success_rate = sum(self.window) / len(self.window)
        if success_rate < self.settings.promote_threshold:
            return

        promoted_to = next_curriculum_k(
            self.state.current_k, self.settings.k_step, self.max_path_length
        )
        self.training_env.env_method("set_reverse_curriculum_k", promoted_to)
        self.state.promotions.append(
            {
                "timesteps": self.model.num_timesteps,
                "from_k": self.state.current_k,
                "to_k": promoted_to,
                "success_rate": round(success_rate, 4),
                "episodes": self.state.episodes,
            }
        )
        logger.info(
            f"Curriculum promoted k={self.state.current_k} -> {promoted_to} at "
            f"{self.model.num_timesteps} steps (success {success_rate:.3f})"
        )
        self.state.current_k = promoted_to
        self.window.clear()

    def _on_rollout_end(self) -> None:
        elapsed = time.perf_counter() - self._started_at
        steps_done = self.model.num_timesteps - self._timesteps_at_start
        row = {
            "timesteps": self.model.num_timesteps,
            "k": self.state.current_k,
            "episodes": self.state.episodes,
            "window_success_rate": self._window_rate(),
            "rollout_episodes": len(self._recent),
            "rollout_success_rate": self._rate(lambda outcome: outcome.solved),
            "rollout_dead_end_rate": self._rate(lambda outcome: outcome.dead_end),
            "rollout_truncated_rate": self._rate(lambda outcome: outcome.truncated),
            "rollout_mean_steps": self._mean(lambda outcome: float(outcome.steps)),
            "rollout_mean_coverage": self._mean(lambda outcome: outcome.coverage),
            "fps": round(steps_done / elapsed, 1) if elapsed > 0 else None,
            "elapsed_s": round(elapsed, 1),
            "gpu_alloc_mib": gpu_mib(th.cuda.max_memory_allocated),
            "gpu_reserved_mib": gpu_mib(th.cuda.max_memory_reserved),
        }
        self._append_jsonl(self.progress_path, row)

        current_k = self.state.current_k
        self.logger.record(
            "curriculum/k", NO_CURRICULUM_TB_VALUE if current_k is None else current_k
        )
        for key in (
            "window_success_rate",
            "rollout_success_rate",
            "rollout_dead_end_rate",
            "rollout_truncated_rate",
        ):
            self.logger.record(f"curriculum/{key}", row[key] or 0.0)
        self._recent.clear()

    def _window_rate(self) -> float | None:
        if not self.window:
            return None
        return round(sum(self.window) / len(self.window), 4)

    def _rate(self, predicate: Callable[[EpisodeOutcome], bool]) -> float | None:
        if not self._recent:
            return None
        matching = sum(1 for outcome in self._recent if predicate(outcome))
        return round(matching / len(self._recent), 4)

    def _mean(self, value: Callable[[EpisodeOutcome], float]) -> float | None:
        if not self._recent:
            return None
        total = sum(value(outcome) for outcome in self._recent)
        return round(total / len(self._recent), 3)

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def sync_state(self) -> CurriculumState:
        self.state.window = list(self.window)
        return self.state


class CheckpointCallback(BaseCallback):
    """Saves the model and the curriculum state together, and never prunes either."""

    def __init__(
        self,
        every_timesteps: int,
        checkpoint_dir: Path,
        state_path: Path,
        curriculum: CurriculumCallback,
        run_config: dict[str, Any],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.every_timesteps = every_timesteps
        self.checkpoint_dir = checkpoint_dir
        self.state_path = state_path
        self.curriculum = curriculum
        self.run_config = run_config
        self._next_at = 0

    def _on_training_start(self) -> None:
        self._next_at = self.model.num_timesteps + self.every_timesteps

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self._next_at:
            self.save(f"model_{self.model.num_timesteps}")
            self._next_at = self.model.num_timesteps + self.every_timesteps
        return True

    def save(self, name: str) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{name}.zip"
        self.model.save(path)
        write_state(
            self.state_path,
            self.curriculum.sync_state(),
            self.model.num_timesteps,
            path,
            self.run_config,
        )
        logger.info(f"Checkpoint {path.name} at {self.model.num_timesteps} steps")
        return path


def gpu_mib(reader: Callable[[], int]) -> float | None:
    if not th.cuda.is_available():
        return None
    return round(reader() / BYTES_PER_MIB, 1)


def write_state(
    path: Path,
    state: CurriculumState,
    num_timesteps: int,
    checkpoint: Path,
    run_config: dict[str, Any],
) -> None:
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "num_timesteps": num_timesteps,
        "checkpoint": str(checkpoint),
        "curriculum": state.to_dict(),
        "config": run_config,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_samples(goal: Goal, split: str) -> list[PuzzleSample]:
    keep = WALL_FILTERS[goal.walls]
    return [
        sample
        for sample in load_split(DATASET_ROOT / goal.dataset, split)
        if sample.puzzle["grid_size"][0] == goal.size and keep(sample)
    ]


def describe_splits(goal: Goal) -> dict[str, int]:
    """Prints what the run will actually train and score on.

    The pickle carries no wall flag, so these counts only exist once the filter has run --
    and a solve rate whose denominator was never written down is not reportable.
    """
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        samples = select_samples(goal, split)
        walled = sum(1 for sample in samples if sample.puzzle["walls"])
        counts[split] = len(samples)
        logger.info(
            f"{split:5s} size={goal.size} walls={goal.walls}: {len(samples):5d} puzzles "
            f"({walled} walled / {len(samples) - walled} wall-free)"
        )
    return counts


def read_action_masks(env: Any) -> np.ndarray:
    """Reads the mask through the wrapper stack (see `make_vec_env`)."""
    return env.unwrapped.action_masks()


def apply_resource_limits(resources: ResourceSettings) -> dict[str, Any]:
    """Caps CPU threads and GPU memory so the machine stays usable during a run.

    Torch's default intra-op thread count is the physical core count, so scaling *that*
    keeps the budget honest on any machine without a psutil dependency.
    """
    threads = max(1, int(th.get_num_threads() * resources.cpu_fraction))
    th.set_num_threads(threads)
    limits = {"torch_threads": threads, "cpu_fraction": resources.cpu_fraction}
    if th.cuda.is_available():
        th.cuda.set_per_process_memory_fraction(resources.gpu_memory_fraction)
        total_mib = th.cuda.get_device_properties(0).total_memory / BYTES_PER_MIB
        limits["gpu_memory_cap_mib"] = round(
            total_mib * resources.gpu_memory_fraction, 1
        )
    logger.info(f"Resource limits: {limits}")
    return limits


def make_vec_env(
    samples: Sequence[PuzzleSample],
    goal: Goal,
    seed: int,
    curriculum_k: int | None,
    vec: str,
) -> VecEnv:
    def factory(rank: int) -> Callable[[], ActionMasker]:
        def _init() -> ActionMasker:
            env = PuzzleEnvV2(
                samples,
                reverse_curriculum_k=curriculum_k,
                shaping_lambda=goal.shaping_lambda,
                gamma=goal.ppo.gamma,
                connectivity_features=goal.connectivity_features,
            )
            env.reset(seed=seed + rank)
            # Gymnasium 1.x dropped attribute pass-through on wrappers, so the mask has
            # to be read from the unwrapped env rather than from the Monitor.
            return ActionMasker(Monitor(env), read_action_masks)

        return _init

    factories = [factory(rank) for rank in range(goal.ppo.n_envs)]
    return SubprocVecEnv(factories) if vec == "subproc" else DummyVecEnv(factories)


def build_model(goal: Goal, vec_env: VecEnv, run_dir: Path, seed: int, device: str):
    ppo = goal.ppo
    return MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        n_epochs=ppo.n_epochs,
        learning_rate=ppo.learning_rate,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        max_grad_norm=ppo.max_grad_norm,
        policy_kwargs={
            "features_extractor_class": GridScalarExtractor,
            "features_extractor_kwargs": {"network": goal.network},
            "normalize_images": False,
        },
        tensorboard_log=str(run_dir / TENSORBOARD_DIRNAME),
        seed=seed,
        device=device,
        verbose=0,
    )


def model_policy(
    model: MaskablePPO,
) -> Callable[[PuzzleEnvV2, np.random.Generator], int]:
    """Wraps a trained model as a `baselines.PolicyFn` so scoring shares one code path."""

    def policy(env: PuzzleEnvV2, rng: np.random.Generator) -> int:
        action, _ = model.predict(
            env.observation(), deterministic=True, action_masks=env.action_masks()
        )
        return int(action)

    return policy


def score(
    model: MaskablePPO,
    samples: Sequence[PuzzleSample],
    seed: int,
    episodes_per_puzzle: int,
    with_baselines: bool,
    connectivity_features: bool = False,
) -> dict[str, Any]:
    """Scores the model and the two controls on the same puzzles, always deterministic.

    The model plays each puzzle once: `deterministic=True` is an argmax and the start cell
    is fixed, so repeats would be identical. The baselines sample, so they need repeats --
    which is also how the 2026-08-15 baseline numbers were measured.
    """
    results = {
        "model": evaluate(
            samples,
            model_policy(model),
            seed=seed,
            episodes_per_puzzle=1,
            label="maskable_ppo",
            connectivity_features=connectivity_features,
        )
    }
    if with_baselines:
        for name in POLICIES:
            results[name] = evaluate(
                samples,
                name,
                seed=seed,
                episodes_per_puzzle=episodes_per_puzzle,
                connectivity_features=connectivity_features,
            )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", choices=sorted(GOALS), required=True)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Overrides the goal's budget; use for pilots and cost measurement.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=None, help="Overrides the goal's PPO n_envs."
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Defaults to the goal key."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Overrides the goal's dataset; use it to reproduce an older run.",
    )
    parser.add_argument(
        "--shaping-lambda",
        type=float,
        default=None,
        help="Overrides the goal's shaping weight; 0 leaves only the terminal +1 and gamma.",
    )
    parser.add_argument(
        "--connectivity-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Adds the unvisited-region component count to the observation. "
        "Changes the observation shape, so it cannot be turned on mid-run.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--vec",
        choices=VEC_ENV_KINDS,
        default=None,
        help="Overrides the goal's vec_env; the goal default is measured, not guessed.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--episode-log", action="store_true")
    return parser


def resolve_goal(args: argparse.Namespace) -> Goal:
    """Applies the per-run overrides on top of the goal definition."""
    goal = GOALS[args.goal]
    if args.timesteps is not None:
        goal = replace(goal, timesteps=args.timesteps)
    if args.n_envs is not None:
        goal = replace(goal, ppo=replace(goal.ppo, n_envs=args.n_envs))
    if args.vec is not None:
        goal = replace(goal, resources=replace(goal.resources, vec_env=args.vec))
    if args.dataset is not None:
        goal = replace(goal, dataset=args.dataset)
    # `is not None` rather than a truth test: 0.0 is the setting the restart plan
    # specifies for the one-stroke phase, and it is the whole point of the override.
    if args.shaping_lambda is not None:
        goal = replace(goal, shaping_lambda=args.shaping_lambda)
    # Same reason as above: `False` is the control arm's explicit setting, not an omission.
    if args.connectivity_features is not None:
        goal = replace(goal, connectivity_features=args.connectivity_features)
    return goal


def main() -> None:
    args = build_arg_parser().parse_args()
    goal = resolve_goal(args)
    run_id = args.run_id or goal.key
    limits = apply_resource_limits(goal.resources)

    run_dir = LOG_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "train.log", level="INFO")
    logger.info(f"Goal {goal.key}: {goal.description}")
    logger.info(
        f"Target solve rate {goal.target_solve_rate:.0%} on {goal.size}x{goal.size}, "
        f"budget {goal.timesteps:,} steps"
    )

    counts = describe_splits(goal)
    train_samples = select_samples(goal, "train")
    eval_samples = select_samples(goal, args.eval_split)
    if not train_samples:
        raise SystemExit(f"No training puzzles for {goal.key}")

    max_path_length = max(len(sample.solution_path) for sample in train_samples)
    logger.info(f"Longest solution in the training split: {max_path_length} cells")

    state_path = run_dir / STATE_FILENAME
    run_config = {
        "goal": asdict(goal),
        "hyperparameters_are_untuned": True,
        "split_counts": counts,
        "max_path_length": max_path_length,
        "seed": args.seed,
        "resource_limits": limits,
    }

    curriculum_state = CurriculumState(current_k=goal.curriculum.k_start)
    resumed_from = None
    if args.resume:
        saved = read_state(state_path)
        curriculum_state = CurriculumState.from_dict(saved["curriculum"])
        resumed_from = Path(saved["checkpoint"])
        logger.info(
            f"Resuming {run_id} from {resumed_from.name} at "
            f"{saved['num_timesteps']} steps, k={curriculum_state.current_k}"
        )

    vec_env = make_vec_env(
        train_samples,
        goal,
        seed=args.seed,
        curriculum_k=curriculum_state.current_k,
        vec=goal.resources.vec_env,
    )
    if resumed_from is not None:
        model = MaskablePPO.load(resumed_from, env=vec_env, device=args.device)
    else:
        model = build_model(goal, vec_env, run_dir, args.seed, args.device)
    parameters = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"Policy parameters: {parameters:,} on {model.device}")

    curriculum = CurriculumCallback(
        state=curriculum_state,
        goal=goal,
        max_path_length=max_path_length,
        progress_path=run_dir / PROGRESS_FILENAME,
        episode_path=run_dir / EPISODE_FILENAME if args.episode_log else None,
    )
    checkpoints = CheckpointCallback(
        every_timesteps=goal.checkpoint_every,
        checkpoint_dir=MODEL_ROOT / run_id / CHECKPOINT_DIRNAME,
        state_path=state_path,
        curriculum=curriculum,
        run_config=run_config,
    )

    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.learn(
        total_timesteps=goal.timesteps,
        callback=[curriculum, checkpoints],
        reset_num_timesteps=not args.resume,
        progress_bar=False,
    )
    training_seconds = time.perf_counter() - started
    checkpoints.save(FINAL_CHECKPOINT_NAME)
    vec_env.close()

    logger.info(
        f"Trained {goal.timesteps:,} steps in {training_seconds:.1f}s "
        f"({goal.timesteps / training_seconds:.0f} fps), "
        f"k={curriculum_state.current_k}, promotions={len(curriculum_state.promotions)}"
    )

    scores = score(
        model,
        eval_samples,
        seed=args.seed,
        episodes_per_puzzle=args.eval_episodes,
        with_baselines=not args.no_baselines,
        connectivity_features=goal.connectivity_features,
    )
    solve_rate = scores["model"]["overall"]["solve_rate"]
    report = {
        "run_id": run_id,
        "config": run_config,
        "training_seconds": round(training_seconds, 1),
        "fps": round(goal.timesteps / training_seconds, 1),
        "gpu_peak_alloc_mib": gpu_mib(th.cuda.max_memory_allocated),
        "gpu_peak_reserved_mib": gpu_mib(th.cuda.max_memory_reserved),
        "final_k": curriculum_state.current_k,
        "promotions": curriculum_state.promotions,
        "eval_split": args.eval_split,
        "eval_puzzles": len(eval_samples),
        "target_solve_rate": goal.target_solve_rate,
        "meets_target": bool(
            solve_rate >= goal.target_solve_rate and curriculum_state.current_k is None
        ),
        "scores": scores,
    }
    (run_dir / f"eval_{args.eval_split}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for name, result in scores.items():
        overall = result["overall"]
        logger.info(
            f"{name:14s} solve={overall['solve_rate']:.3f} "
            f"dead_end={overall['dead_end_rate']:.3f} "
            f"coverage={overall['mean_coverage']:.3f} (n={overall['episodes']})"
        )
    logger.info(
        f"Done condition (solve >= {goal.target_solve_rate:.0%} at full length): "
        f"{'MET' if report['meets_target'] else 'not met'}"
    )


if __name__ == "__main__":
    main()
