# src/core/tests/rl/test_train_behaviour_cloning.py
"""Unit tests for the supervised warm start.

Three of these pin things that would be wrong *silently*:

*   A replay that drifts by one step still yields plausible-looking pairs -- the label
    would just belong to the previous state. `test_replayed_labels_solve_the_puzzle` ties
    the labels to the env's own definition of solved, so a drift breaks the test rather
    than quietly training on shifted data.
*   `choice_accuracy` exists because an accuracy over *all* decisions mostly measures how
    forced the board is: 69.1% of 4x4 decisions have a single legal move, so an untrained
    policy already scores ~0.69. A regression that started reporting the unfiltered number
    would look like a large improvement.
*   Behaviour cloning is a third place an env gets constructed, after training and
    evaluation. Trap #24 cost three experiment arms because a flag reached only one of
    them, so the observation spaces are pinned to agree here too.

Nothing here loads a pickled dataset: `datasets/` is not in version control.
"""

import random

import numpy as np
import pytest

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.baselines import make_eval_env
from src.core.rl.rl_env_v2 import SUCCESS_REWARD, PuzzleSample
from src.core.rl.train_behaviour_cloning import (
    MIN_LEGAL_ACTIONS_FOR_A_CHOICE,
    _batches,
    _stack,
    iter_supervised_pairs,
)

GENERATOR_SEED = 20260815
BOARD_SIZE = 4


@pytest.fixture(scope="module")
def sample() -> PuzzleSample:
    random.seed(GENERATOR_SEED)
    result = generate_puzzle(
        m=BOARD_SIZE, n=BOARD_SIZE, has_walls=True, timeout_per_attempt=5.0
    )
    assert result is not None, "generator returned None; retry with another seed"
    puzzle, solution_path = result
    return PuzzleSample(puzzle=puzzle, solution_path=solution_path)


def test_replayed_labels_solve_the_puzzle(sample: PuzzleSample) -> None:
    """The labels are the solution: replaying them has to end in a solved board."""
    pairs = list(iter_supervised_pairs([sample], connectivity_features=False))

    env = make_eval_env(sample)
    env.reset(seed=GENERATOR_SEED)
    info = {"solved": False}
    for pair in pairs:
        _, _, terminated, _, info = env.step(pair.action)

    assert info["solved"], "replaying the labelled actions did not solve the puzzle"
    assert terminated
    assert len(pairs) == len(sample.solution_path) - 1


def test_every_label_is_legal_in_the_state_it_is_paired_with(
    sample: PuzzleSample,
) -> None:
    """Catches a one-step drift between the observation and the action stored with it."""
    for pair in iter_supervised_pairs([sample], connectivity_features=False):
        assert pair.action_mask[
            pair.action
        ], f"label {pair.action} is masked out in the state it is paired with"


def test_masks_are_snapshots_not_live_views(sample: PuzzleSample) -> None:
    """The env mutates in place, so a mask kept by reference would all end up identical."""
    masks = [
        pair.action_mask
        for pair in iter_supervised_pairs([sample], connectivity_features=False)
    ]
    assert (
        len({mask.tobytes() for mask in masks}) > 1
    ), "every mask is identical; they are probably views onto the same array"


@pytest.mark.parametrize("connectivity_features", [False, True])
def test_cloning_observations_match_the_evaluation_space(
    sample: PuzzleSample, connectivity_features: bool
) -> None:
    """Behaviour cloning is a third env construction point; trap #24 was the first two."""
    env = make_eval_env(sample, connectivity_features=connectivity_features)
    pairs = list(iter_supervised_pairs([sample], connectivity_features))

    for key, space in env.observation_space.spaces.items():
        assert pairs[0].observation[key].shape == space.shape


def test_choice_states_exclude_forced_moves(sample: PuzzleSample) -> None:
    """`choice_accuracy` must not be inflated by states where the mask leaves one option.

    On 4x4 about 69% of decisions are forced, so an accuracy that counts them is mostly a
    measure of the board, not of the policy.
    """
    masks = np.stack(
        [
            pair.action_mask
            for pair in iter_supervised_pairs([sample], connectivity_features=False)
        ]
    )
    is_choice = masks.sum(axis=1) >= MIN_LEGAL_ACTIONS_FOR_A_CHOICE

    assert is_choice.sum() < len(masks), "expected some forced moves on a 4x4 board"
    assert masks[~is_choice].sum(axis=1).max(initial=0) <= 1


def test_batches_cover_every_pair_exactly_once(sample: PuzzleSample) -> None:
    pairs = list(iter_supervised_pairs([sample], connectivity_features=False))
    batched = [pair for batch in _batches(iter(pairs), 4) for pair in batch]

    assert len(batched) == len(pairs)
    assert [pair.action for pair in batched] == [pair.action for pair in pairs]


def test_stack_keeps_the_batch_aligned(sample: PuzzleSample) -> None:
    pairs = list(iter_supervised_pairs([sample], connectivity_features=False))[:3]
    observation, masks, actions, returns = _stack(pairs)

    assert actions.tolist() == [pair.action for pair in pairs]
    # float32 on purpose: the batch feeds torch, so exact equality with the Python float
    # the replay accumulated would be testing the dtype rather than the alignment.
    assert returns.dtype == np.float32
    assert returns.tolist() == pytest.approx([pair.value_target for pair in pairs])
    assert masks.shape == (len(pairs), len(pairs[0].action_mask))
    for key in observation:
        assert observation[key].shape[0] == len(pairs)
        assert np.array_equal(observation[key][1], pairs[1].observation[key])


def test_value_targets_are_the_discounted_return_of_the_replay(
    sample: PuzzleSample,
) -> None:
    """The critic's target has to be the return the fine-tuning env actually pays out.

    Success is worth `SUCCESS_REWARD` on the last step, so with no shaping the target is
    exactly `gamma ** steps_remaining` -- which also pins the discounting direction: a
    backwards accumulation that ran forwards would put the largest value at the start.
    """
    gamma = 0.9
    pairs = list(
        iter_supervised_pairs(
            [sample], connectivity_features=False, shaping_lambda=0.0, gamma=gamma
        )
    )

    for index, pair in enumerate(pairs):
        remaining = len(pairs) - 1 - index
        assert pair.value_target == pytest.approx(
            SUCCESS_REWARD * gamma**remaining, rel=1e-6
        )
    assert pairs[-1].value_target > pairs[0].value_target


def test_shaping_shows_up_in_the_value_target(sample: PuzzleSample) -> None:
    """Shaping is part of the reward the fine-tune pays, so it must be in the target."""
    plain = list(
        iter_supervised_pairs([sample], connectivity_features=False, shaping_lambda=0.0)
    )
    shaped = list(
        iter_supervised_pairs([sample], connectivity_features=False, shaping_lambda=0.2)
    )

    assert [p.action for p in plain] == [p.action for p in shaped]
    assert shaped[0].value_target > plain[0].value_target
