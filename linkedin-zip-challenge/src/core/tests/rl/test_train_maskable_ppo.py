# src/core/tests/rl/test_train_maskable_ppo.py
"""A2 unit tests for the MaskablePPO training harness.

Two of these pin defects that cost the track real time and would not show up as a crash:

*   SB3 silently flattens our observation instead of convolving it, because the grid is
    `float32` in 0-1 rather than `uint8` in 0-255. `test_sb3_would_flatten_the_grid`
    documents that fact, so a future "why do we ship a custom extractor?" has an answer.
*   A resume that restores the model but not `reverse_curriculum_k` restarts the
    curriculum while every curve still looks healthy.

Nothing here loads a pickled dataset: `datasets/` is not in version control.
"""

import os
import random

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.rl_env_v2 import PuzzleEnvV2, PuzzleSample
from src.core.rl.train_config import (
    GOALS,
    Goal,
    NetworkSettings,
    ResourceSettings,
    capped_worker_count,
)
from src.core.rl.train_maskable_ppo import (
    WALL_FILTERS,
    apply_resource_limits,
    CurriculumState,
    EpisodeOutcome,
    GridScalarExtractor,
    classify_episode,
    next_curriculum_k,
    read_action_masks,
    resolve_goal,
)
from src.core.utils import Puzzle

GENERATOR_SEED = 20260829
BATCH_SIZE = 3


@pytest.fixture(scope="module")
def sample() -> PuzzleSample:
    random.seed(GENERATOR_SEED)
    result = generate_puzzle(m=4, n=4, has_walls=True, timeout_per_attempt=5.0)
    assert result is not None, "generator returned None; retry with another seed"
    puzzle, solution_path = result
    return PuzzleSample(puzzle=puzzle, solution_path=solution_path)


def _observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(8, 8, 8), dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
        }
    )


def test_sb3_would_flatten_the_grid() -> None:
    """The reason a custom extractor exists: SB3 does not see our grid as an image."""
    assert is_image_space(_observation_space()["grid"]) is False


def test_extractor_returns_the_configured_feature_width() -> None:
    space = _observation_space()
    network = NetworkSettings(conv_channels=8, features_dim=32)
    extractor = GridScalarExtractor(space, network=network)

    features = extractor(
        {
            "grid": th.zeros((BATCH_SIZE, *space["grid"].shape)),
            "scalars": th.zeros((BATCH_SIZE, *space["scalars"].shape)),
        }
    )

    assert features.shape == (BATCH_SIZE, network.features_dim)


def test_extractor_keeps_the_board_geometry() -> None:
    """Two boards differing only in cell position must not collapse to equal features."""
    space = _observation_space()
    extractor = GridScalarExtractor(space, network=NetworkSettings(conv_channels=8))
    scalars = th.zeros((1, *space["scalars"].shape))

    left = th.zeros((1, *space["grid"].shape))
    left[0, 0, 0, 0] = 1.0
    right = th.zeros((1, *space["grid"].shape))
    right[0, 0, 7, 7] = 1.0

    assert not th.allclose(
        extractor({"grid": left, "scalars": scalars}),
        extractor({"grid": right, "scalars": scalars}),
    )


def test_read_action_masks_reaches_through_wrappers(sample: PuzzleSample) -> None:
    """Gymnasium 1.x dropped attribute pass-through; the mask must still be reachable."""
    from stable_baselines3.common.monitor import Monitor

    env = PuzzleEnvV2([sample], reverse_curriculum_k=3)
    env.reset(seed=GENERATOR_SEED)
    wrapped = Monitor(env)

    assert np.array_equal(read_action_masks(wrapped), env.action_masks())


def test_observation_is_publicly_readable(sample: PuzzleSample) -> None:
    """Scoring a model through `baselines.PolicyFn` needs the observation without `_`."""
    env = PuzzleEnvV2([sample], reverse_curriculum_k=3)
    observation, _ = env.reset(seed=GENERATOR_SEED)

    published = env.observation()

    assert np.array_equal(published["grid"], observation["grid"])
    assert np.array_equal(published["scalars"], observation["scalars"])


@pytest.mark.parametrize(
    ("current_k", "expected"),
    [(3, 6), (6, 9), (12, 15), (13, None), (15, None)],
)
def test_next_curriculum_k(current_k: int, expected: int | None) -> None:
    assert next_curriculum_k(current_k, k_step=3, max_path_length=16) == expected


def test_episode_buckets_are_mutually_exclusive() -> None:
    solved = classify_episode({"solved": True, "dead_end": False, "steps": 9})
    stuck = classify_episode({"solved": False, "dead_end": True, "steps": 4})
    timed_out = classify_episode({"solved": False, "dead_end": False, "steps": 16})

    for outcome in (solved, stuck, timed_out):
        assert [outcome.solved, outcome.dead_end, outcome.truncated].count(True) == 1
    assert timed_out.truncated is True


def test_curriculum_state_round_trip() -> None:
    """A resume reads this back; losing `current_k` silently restarts the curriculum."""
    state = CurriculumState(
        current_k=9,
        window=[True, False, True],
        promotions=[{"from_k": 6, "to_k": 9}],
        episodes=1858,
    )

    restored = CurriculumState.from_dict(state.to_dict())

    assert restored == state


def test_resolved_goal_overrides_do_not_touch_the_registry() -> None:
    class Args:
        goal = "goal1_4x4"
        timesteps = 4096
        n_envs = 2
        vec = "subproc"

    resolved = resolve_goal(Args())

    assert resolved.timesteps == Args.timesteps
    assert resolved.ppo.n_envs == Args.n_envs
    assert resolved.resources.vec_env == Args.vec
    registered = GOALS["goal1_4x4"]
    assert registered.timesteps != Args.timesteps
    assert registered.ppo.n_envs != Args.n_envs
    assert registered.resources.vec_env == "dummy"


def test_worker_count_leaves_cores_free() -> None:
    """Generation used to run an unbounded pool and pegged every core; it must not again."""
    cores = os.cpu_count() or 1
    assert 1 <= capped_worker_count(0.75) <= max(1, int(cores * 0.75))
    assert capped_worker_count(0.75) < cores or cores == 1


def test_resource_limits_stay_under_the_configured_fraction() -> None:
    """The point of the cap is that the machine stays usable, so it must actually bind."""
    default_threads = th.get_num_threads()
    try:
        limits = apply_resource_limits(ResourceSettings(cpu_fraction=0.75))
        assert limits["torch_threads"] <= default_threads * 0.75 + 1
        assert th.get_num_threads() == limits["torch_threads"]
    finally:
        th.set_num_threads(default_threads)


def test_every_goal_targets_a_rate_above_its_baselines() -> None:
    """Measured 2026-08-15: greedy peaks at 10.2% on 4x4 and 0.8% on 6x6."""
    for goal in GOALS.values():
        assert 0.0 < goal.target_solve_rate <= 1.0
        assert goal.target_solve_rate > 0.5


def test_goal_rejects_an_unknown_wall_policy() -> None:
    with pytest.raises(ValueError, match="walls must be one of"):
        Goal(
            key="broken",
            size=6,
            description="",
            target_solve_rate=0.85,
            timesteps=1,
            walls="sometimes",
        )


def test_wall_filters_partition_the_samples(sample: PuzzleSample) -> None:
    walled = PuzzleSample(
        puzzle={**sample.puzzle, "walls": {((0, 0), (0, 1))}},
        solution_path=sample.solution_path,
    )
    bare: Puzzle = {**sample.puzzle, "walls": set()}
    wall_free = PuzzleSample(puzzle=bare, solution_path=sample.solution_path)

    assert WALL_FILTERS["only"](walled) and not WALL_FILTERS["only"](wall_free)
    assert WALL_FILTERS["none"](wall_free) and not WALL_FILTERS["none"](walled)
    assert WALL_FILTERS["all"](walled) and WALL_FILTERS["all"](wall_free)


def test_episode_outcome_defaults_to_zero_when_info_is_empty() -> None:
    assert classify_episode({}) == EpisodeOutcome(
        solved=False, dead_end=False, truncated=True, steps=0, coverage=0.0
    )
