# src/core/rl/train_config.py
"""The single place where a training target and its knobs are defined.

`GOALS` is the one thing to edit when the plan changes: a goal names the boards it trains
on, how long it runs and the solve rate it has to clear to count as done. Everything else
in the training script reads from here, so no board size, wall policy or hyperparameter is
written down twice.

The hyperparameter defaults are the **untuned** starting values from the RL restart plan
section 4.8. No sensitivity study has been run, so treat any result as "this setting", not
"the best setting".
"""

import os
from dataclasses import dataclass, field

# A2 (2026-08-29) was measured on `main_n1700_456`, 1,360 training puzzles per size, and
# showed both goals memorising it. This pack is ~11x larger and deduplicated, which is the
# experiment handover section 6 asks for. To reproduce an A2 number, pass
# `--dataset main_n1700_456`.
DEFAULT_DATASET = "seed20300000_n20000_4-6"

# Leave the machine usable while anything here is running. This is the *one* place the
# budget is defined: it caps PyTorch's threads during training and the worker pool during
# dataset generation, which was previously unbounded and pegged all 24 logical cores.
DEFAULT_CPU_FRACTION = 0.75


# The parent process works too (task feeding, progress, result collection) and the OS takes
# its cut, so a worker per budgeted core overshoots: measured 2026-08-29 on 24 logical cores,
# 18 workers held system CPU at 74-82% against a 75% target. Two are given back for that.
WORKERS_RESERVED_FOR_PARENT = 2


def capped_worker_count(fraction: float = DEFAULT_CPU_FRACTION) -> int:
    """Worker processes to run so that generation leaves headroom for everything else."""
    budgeted = int((os.cpu_count() or 1) * fraction)
    return max(1, budgeted - WORKERS_RESERVED_FOR_PARENT)


# The generator only ever emits 0 or 2-5 walls, so "all" is a mixed set, not a hard one.
WALL_POLICIES = ("all", "none", "only")
VEC_ENV_KINDS = ("dummy", "subproc")


@dataclass(frozen=True)
class PPOSettings:
    """Restart plan section 4.8 starting values. Untuned."""

    n_envs: int = 16
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class NetworkSettings:
    """Three padded 3x3 convolutions, no pooling (restart plan section 4.7)."""

    conv_channels: int = 64
    conv_kernel: int = 3
    conv_padding: int = 1
    features_dim: int = 256


@dataclass(frozen=True)
class CurriculumSettings:
    """Reverse curriculum: start `k` cells from the end, advance on measured success."""

    k_start: int = 3
    k_step: int = 3
    promote_threshold: float = 0.9
    promote_window: int = 200


@dataclass(frozen=True)
class ResourceSettings:
    """Headroom, so the machine stays usable while a run is going.

    `cpu_fraction` scales PyTorch's *own* default intra-op thread count. Torch sets that
    to the physical core count (12 of 24 logical on this machine), so multiplying it keeps
    the budget meaningful without guessing at hyperthreading or importing psutil, which is
    only present here as a transitive dependency of another track.

    `vec_env` is "dummy" on measurement rather than preference: `SubprocVecEnv` was slower
    on this environment (3,221 vs 3,880 fps over 100k steps, 2026-08-29) because a step is
    cheap enough that Windows process IPC costs more than the parallelism returns.
    """

    cpu_fraction: float = DEFAULT_CPU_FRACTION
    gpu_memory_fraction: float = 0.75
    vec_env: str = "dummy"


@dataclass(frozen=True)
class Goal:
    """One training target: the board it learns on and the bar it has to clear."""

    key: str
    size: int
    description: str
    target_solve_rate: float
    timesteps: int
    walls: str = "all"
    dataset: str = DEFAULT_DATASET
    shaping_lambda: float = 0.2
    # Checkpoints are never pruned (the VLM track lost an experiment to
    # `save_total_limit=2`), so the interval -- not a retention limit -- is what keeps the
    # run off the disk: one checkpoint is 14 MB, so aim for 25-50 per run.
    checkpoint_every: int = 40_000
    ppo: PPOSettings = field(default_factory=PPOSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    curriculum: CurriculumSettings = field(default_factory=CurriculumSettings)
    resources: ResourceSettings = field(default_factory=ResourceSettings)

    def __post_init__(self) -> None:
        if self.walls not in WALL_POLICIES:
            raise ValueError(
                f"walls must be one of {WALL_POLICIES}, got {self.walls!r}"
            )
        if self.resources.vec_env not in VEC_ENV_KINDS:
            raise ValueError(
                f"vec_env must be one of {VEC_ENV_KINDS}, "
                f"got {self.resources.vec_env!r}"
            )


# Measured baselines on the held-out split (logs/rl_baselines/, 2026-08-15):
#   4x4  masked random 8.8%   greedy 10.2%
#   6x6  masked random 0.0%   greedy  0.8%
# The targets below come from the track plan: 90% for stage A2, 85% once walls and the
# larger board are in play.
GOALS: dict[str, Goal] = {
    "goal1_4x4": Goal(
        key="goal1_4x4",
        size=4,
        description="Known-good anchor: if this fails, the machinery is wrong, not the task.",
        target_solve_rate=0.90,
        timesteps=1_000_000,
    ),
    "goal2_6x6": Goal(
        key="goal2_6x6",
        size=6,
        description="The product-relevant board; LinkedIn Zip puzzles are 6x6.",
        target_solve_rate=0.85,
        timesteps=5_000_000,
        checkpoint_every=100_000,
    ),
}
