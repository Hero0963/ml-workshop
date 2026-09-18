# src/core/tests/rl/test_collect_solutions.py
"""Unit tests for Expert Iteration's collection step and the relabelling that consumes it.

What would go wrong *silently* here is a bad label: a walk that the collector believes
solves the puzzle but does not, or a relabel that hands the network a path belonging to a
different puzzle. Both would train without an error and show up only as a worse model.
So every kept walk is checked twice (env replay and the independent scorer), and the
relabel is pinned to keep each puzzle's own solutions.

Nothing here loads a pickled dataset: `datasets/` is not in version control.
"""

import random
from dataclasses import replace

import numpy as np
import pytest

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.collect_solutions import (
    alternative_walks,
    ends_on_last_number,
    is_valid_solution,
    sample_actions,
    sample_key,
    sample_policy_walks,
)
from src.core.rl.rl_env_v2 import PuzzleSample
from src.core.rl.train_behaviour_cloning import iter_supervised_pairs, relabel
from src.core.rl.train_config import GOALS
from src.core.rl.train_maskable_ppo import build_model, make_vec_env

GENERATOR_SEED = 20260815
BOARD_SIZE = 4
ATTEMPTS = 64


@pytest.fixture(scope="module")
def sample() -> PuzzleSample:
    random.seed(GENERATOR_SEED)
    result = generate_puzzle(
        m=BOARD_SIZE, n=BOARD_SIZE, has_walls=True, timeout_per_attempt=5.0
    )
    assert result is not None, "generator returned None; retry with another seed"
    puzzle, solution_path = result
    return PuzzleSample(puzzle=puzzle, solution_path=solution_path)


@pytest.fixture(scope="module")
def untrained_model(sample: PuzzleSample, tmp_path_factory):
    base = GOALS["goal1_4x4"]
    goal = replace(base, ppo=replace(base.ppo, n_envs=1))
    vec_env = make_vec_env(
        [sample], goal, seed=GENERATOR_SEED, curriculum_k=None, vec="dummy"
    )
    run_dir = tmp_path_factory.mktemp("collector")
    return build_model(goal, vec_env, run_dir, seed=GENERATOR_SEED, device="cpu")


def test_the_checker_accepts_the_dataset_solution_and_rejects_a_broken_one(
    sample: PuzzleSample,
) -> None:
    assert is_valid_solution(sample, sample.solution_path)
    assert not is_valid_solution(sample, sample.solution_path[:-1])
    assert not is_valid_solution(sample, list(reversed(sample.solution_path)))


def test_sampled_actions_are_always_legal() -> None:
    rng = np.random.default_rng(GENERATOR_SEED)
    masks = rng.random((5000, 4)) < 0.5
    masks[~masks.any(axis=1), 0] = True
    # Leftover mass on illegal actions, as float32 softmax leaves it.
    probs = rng.random((5000, 4)) + np.where(masks, 0.0, 1e-9)
    probs /= probs.sum(axis=1, keepdims=True)

    actions = sample_actions(probs, masks, rng)

    assert masks[np.arange(len(actions)), actions].all()


def test_sampled_actions_follow_the_distribution() -> None:
    rng = np.random.default_rng(GENERATOR_SEED)
    draws = 200_000
    probs = np.tile([0.1, 0.0, 0.6, 0.3], (draws, 1))
    masks = np.tile([True, False, True, True], (draws, 1))

    counts = np.bincount(sample_actions(probs, masks, rng), minlength=4) / draws

    assert counts == pytest.approx([0.1, 0.0, 0.6, 0.3], abs=0.01)


def test_every_collected_walk_really_solves_the_puzzle(
    sample: PuzzleSample, untrained_model
) -> None:
    """Each kept walk has to pass the env replay *and* the independent scorer."""
    rng = np.random.default_rng(GENERATOR_SEED)
    [outcome] = sample_policy_walks(
        untrained_model, [sample], ATTEMPTS, rng, parallel=16
    )

    assert outcome.attempts == ATTEMPTS
    assert outcome.rejected_by_checker == 0
    assert outcome.solved_attempts > 0, "no solve at all; the test would pass vacuously"
    assert outcome.walks
    for walk in outcome.walks:
        assert is_valid_solution(sample, walk)
        replayed = sample._replace(solution_path=list(walk))
        pairs = list(iter_supervised_pairs([replayed], connectivity_features=False))
        assert len(pairs) == len(walk) - 1


def test_kept_walks_end_on_the_highest_number(
    sample: PuzzleSample, untrained_model
) -> None:
    """The judges accept walks that pass the last number and keep going; labels must not."""
    rng = np.random.default_rng(GENERATOR_SEED)
    [outcome] = sample_policy_walks(
        untrained_model, [sample], ATTEMPTS, rng, parallel=16
    )

    assert all(ends_on_last_number(sample, walk) for walk in outcome.walks)
    assert not any(
        ends_on_last_number(sample, walk) for walk in outcome.off_last_number
    )
    assert ends_on_last_number(sample, sample.solution_path)
    assert not ends_on_last_number(sample, sample.solution_path[:-1])


def test_alternatives_exclude_the_dataset_solution(
    sample: PuzzleSample, untrained_model
) -> None:
    rng = np.random.default_rng(GENERATOR_SEED)
    [outcome] = sample_policy_walks(
        untrained_model, [sample], ATTEMPTS, rng, parallel=16
    )
    outcome.walks.add(tuple(tuple(cell) for cell in sample.solution_path))

    alternatives = alternative_walks(sample, outcome)

    assert tuple(tuple(cell) for cell in sample.solution_path) not in alternatives
    assert len(alternatives) == len(outcome.walks) - 1


def test_relabel_draws_from_the_puzzles_own_solutions(sample: PuzzleSample) -> None:
    other_walk = [(0, 0), (0, 1)]
    extra = {sample_key(sample): [other_walk]}
    rng = random.Random(GENERATOR_SEED)

    labels = {tuple(relabel([sample], extra, rng)[0].solution_path) for _ in range(200)}

    assert labels == {tuple(sample.solution_path), tuple(other_walk)}


def test_relabel_leaves_puzzles_without_alternatives_alone(
    sample: PuzzleSample,
) -> None:
    rng = random.Random(GENERATOR_SEED)

    [relabelled] = relabel([sample], {}, rng)

    assert relabelled is sample


def test_relabel_keeps_the_puzzle_and_the_order(sample: PuzzleSample) -> None:
    other = sample._replace(puzzle={**sample.puzzle, "grid_size": (9, 9)})
    extra = {sample_key(sample): [[(0, 0)]]}

    relabelled = relabel([sample, other, sample], extra, random.Random(1))

    assert [s.puzzle["grid_size"] for s in relabelled] == [
        sample.puzzle["grid_size"],
        (9, 9),
        sample.puzzle["grid_size"],
    ]
    assert relabelled[1] is other
