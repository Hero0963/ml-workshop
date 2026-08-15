# src/core/tests/rl/test_rl_env_v1_diagnosis.py
"""A0 environment-sanity checks for the legacy `PuzzleEnv` (src/core/rl/rl_env.py).

Two questions answered with experiments instead of code reading:

1. Can the environment be solved *at all*? Replaying a ground-truth solution must
   terminate with the success reward -- if it cannot, no amount of training helps.
   **It cannot**: `reset()` leaves `_next_waypoint_idx` at 0 while the agent already
   stands on waypoint 1, so the first target is the start cell itself and the index
   only advances if the agent steps *back* onto it. A legal one-stroke path never
   returns, so `terminated` is unreachable. Those replays are marked `xfail(strict)`:
   they pin the defect, and they will fail loudly if v1 is ever fixed.
2. Does the observation space really contain the 2-cycle that the 2026-08-15 RL
   restart plan blames for the deterministic policy loop? **Yes** -- confirmed below.

v1 is kept unmodified as the control for env v2; the fixes belong in v2.

Full evidence dump: `uv run python -m src.core.rl.diagnose_env_v1`.
"""

import random

import numpy as np
import pytest
from loguru import logger

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.action_space import ACTION_DELTAS, path_to_actions
from src.core.rl.rl_env import PuzzleEnv
from src.core.tests.conftest import puzzles_to_test
from src.core.utils import Puzzle

SUCCESS_REWARD_THRESHOLD = 900.0
# The generator picks a random start cell, and on odd-sized open grids half of the
# cells have the wrong parity to host a Hamiltonian path, so a seed can exhaust all
# retries. This value is verified to produce a puzzle for every case below.
GENERATOR_SEED = 42
OSCILLATION_ROUND_TRIPS = 4
WALL_BUMPS_BEYOND_BUDGET = 10
UNREACHABLE_TERMINAL_REASON = (
    "env v1 defect: reset() targets the start cell as waypoint 1, so a ground-truth "
    "one-stroke path never advances _next_waypoint_idx and never terminates"
)

# The reference puzzle for the cycle probes: 6x6, no walls, hand-verified solution.
REFERENCE_PUZZLE: Puzzle = puzzles_to_test[1][0]
REFERENCE_SOLUTION: list[tuple[int, int]] = puzzles_to_test[1][1]


def _replay(env: PuzzleEnv, solution_path: list[tuple[int, int]]) -> dict[str, object]:
    """Feeds a ground-truth path into the env and reports how the episode ended."""
    env.reset()
    terminated = truncated = False
    reward = 0.0
    for step_index, action in enumerate(path_to_actions(solution_path)):
        _, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            logger.info(
                f"Episode ended at step {step_index + 1}/{len(solution_path) - 1} "
                f"(terminated={terminated}, truncated={truncated})"
            )
            break
    return {
        "terminated": terminated,
        "truncated": truncated,
        "final_reward": reward,
        "next_waypoint_idx": env._next_waypoint_idx,
        "waypoint_count": len(env.waypoints),
        "visited_cells": len(set(env.path_taken)),
        "visitable_cells": env.total_visitable_cells,
    }


def _assert_ground_truth_solves(
    puzzle: Puzzle, solution_path: list[tuple[int, int]], puzzle_id: str
) -> None:
    outcome = _replay(PuzzleEnv(puzzle), solution_path)
    logger.info(f"{puzzle_id} replay outcome: {outcome}")

    assert outcome["visited_cells"] == outcome["visitable_cells"], (
        f"{puzzle_id}: the ground-truth path did not cover every visitable cell "
        f"({outcome['visited_cells']}/{outcome['visitable_cells']}) -- the replay itself is wrong."
    )
    assert not outcome[
        "truncated"
    ], f"{puzzle_id}: episode was truncated before the path ended."
    assert outcome["terminated"], (
        f"{puzzle_id}: replaying the ground-truth solution did NOT terminate. "
        f"Waypoint progress stalled at index {outcome['next_waypoint_idx']} "
        f"of {outcome['waypoint_count']}."
    )
    assert outcome["final_reward"] >= SUCCESS_REWARD_THRESHOLD, (
        f"{puzzle_id}: terminated without the success bonus (final reward "
        f"{outcome['final_reward']})."
    )


@pytest.mark.xfail(strict=True, reason=UNREACHABLE_TERMINAL_REASON)
@pytest.mark.parametrize(
    "puzzle_data, expected_solution, puzzle_id",
    puzzles_to_test,
    ids=[p[2] for p in puzzles_to_test],
)
def test_ground_truth_replay_terminates_on_fixture_puzzles(
    puzzle_data: Puzzle, expected_solution: list[tuple[int, int]], puzzle_id: str
) -> None:
    """The hand-verified puzzles from conftest must be solvable inside the env."""
    _assert_ground_truth_solves(puzzle_data, expected_solution, puzzle_id)


@pytest.mark.xfail(strict=True, reason=UNREACHABLE_TERMINAL_REASON)
@pytest.mark.parametrize(
    "rows, cols, has_walls",
    [(4, 4, False), (5, 5, True)],
    ids=["generated_4x4_open", "generated_5x5_walls"],
)
def test_ground_truth_replay_terminates_on_generated_puzzles(
    rows: int, cols: int, has_walls: bool
) -> None:
    """`generate_puzzle` returns (puzzle, solution); that solution must solve the env."""
    random.seed(GENERATOR_SEED)
    result = generate_puzzle(m=rows, n=cols, has_walls=has_walls, num_blocked_cells=0)

    assert result is not None, "Puzzle generation failed; cannot run the replay check."
    puzzle, solution_path = result
    _assert_ground_truth_solves(puzzle, solution_path, f"generated_{rows}x{cols}")


def test_first_waypoint_is_only_credited_by_re_entering_the_start_cell() -> None:
    """Pins the root cause of the unreachable terminal state found above."""
    env = PuzzleEnv(REFERENCE_PUZZLE)
    env.reset()

    assert (
        env.start_pos == env.waypoints[env._next_waypoint_idx]
    ), "reset() is expected to aim at the cell the agent already occupies."

    leave_action = path_to_actions(REFERENCE_SOLUTION[:2])[0]
    env.step(leave_action)
    assert env._next_waypoint_idx == 0, "Leaving the start must not credit a waypoint."

    return_action = path_to_actions(REFERENCE_SOLUTION[1::-1])[0]
    _, reward, _, _, _ = env.step(return_action)

    logger.info(
        f"Stepping back onto the start cell {env.start_pos} advanced the waypoint index "
        f"to {env._next_waypoint_idx} with reward {reward}"
    )
    assert env._next_waypoint_idx == 1, (
        "Waypoint 1 is credited only by re-entering the start cell -- the defect that "
        "makes a legal one-stroke path unable to finish."
    )


@pytest.mark.parametrize(
    "puzzle_data, expected_solution, puzzle_id",
    puzzles_to_test,
    ids=[p[2] for p in puzzles_to_test],
)
def test_success_is_reserved_for_paths_that_revisit_the_start_cell(
    puzzle_data: Puzzle, expected_solution: list[tuple[int, int]], puzzle_id: str
) -> None:
    """The same solution terminates once an illegal start-cell detour is prepended.

    Legal Zip solutions score around -35 and never finish (see the xfailed replays);
    prefixing a single step off and back onto the start collects waypoint 1 and the
    episode ends with the +1000 bonus. env v1's reward is therefore anti-correlated
    with the rules of the game it is supposed to teach.
    """
    detour_path = [expected_solution[0], expected_solution[1], *expected_solution]
    assert len(set(detour_path)) < len(detour_path), "The detour must revisit a cell."

    outcome = _replay(PuzzleEnv(puzzle_data), detour_path)
    logger.info(f"{puzzle_id} detour replay outcome: {outcome}")

    assert outcome["terminated"], (
        f"{puzzle_id}: even the start-cell detour failed to terminate; the diagnosis "
        "of the v1 defect is incomplete."
    )
    assert (
        outcome["final_reward"] >= SUCCESS_REWARD_THRESHOLD
    ), f"{puzzle_id}: terminated without the success bonus ({outcome['final_reward']})."


def _obs_hash(observation: np.ndarray) -> int:
    return hash(observation.tobytes())


def test_observation_forms_a_two_cycle_when_oscillating() -> None:
    """Bouncing between two visited cells must yield only two distinct observations.

    This is the mechanism blamed for the 2025-10 deterministic policy loop: the path
    channel is binary and the waypoint index does not advance, so the observation
    sequence is o_A, o_B, o_A, ... and a deterministic policy has a fixed point.
    """
    env = PuzzleEnv(REFERENCE_PUZZLE)
    env.reset()
    for action in path_to_actions(REFERENCE_SOLUTION[:3]):
        env.step(action)

    cell_a, cell_b = REFERENCE_SOLUTION[2], REFERENCE_SOLUTION[1]
    action_to_b = path_to_actions([cell_a, cell_b])[0]
    action_to_a = path_to_actions([cell_b, cell_a])[0]

    observations: list[int] = []
    rewards: list[float] = []
    for _ in range(OSCILLATION_ROUND_TRIPS):
        for action in (action_to_b, action_to_a):
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(_obs_hash(observation))
            rewards.append(reward)
            assert (
                not terminated
            ), "Oscillating between visited cells must not terminate."
            assert (
                not truncated
            ), "Oscillation exceeded the step budget; shorten the probe."

    distinct = set(observations)
    logger.info(
        f"Oscillation between {cell_a} and {cell_b}: {len(distinct)} distinct "
        f"observations over {len(observations)} steps, rewards={rewards}"
    )
    assert len(distinct) == 2, (
        f"Expected exactly two distinct observations while oscillating, got {len(distinct)}. "
        "The 2-cycle hypothesis behind the RL redesign would be refuted."
    )
    assert observations[::2] == [observations[0]] * OSCILLATION_ROUND_TRIPS
    assert observations[1::2] == [observations[1]] * OSCILLATION_ROUND_TRIPS


def test_illegal_moves_freeze_the_observation_and_never_truncate() -> None:
    """Bumping a boundary is a degenerate 1-cycle, and it does not consume the budget."""
    env = PuzzleEnv(REFERENCE_PUZZLE)
    env.reset()

    row, col = env._agent_location
    illegal_action = next(
        action
        for action, (delta_row, delta_col) in ACTION_DELTAS.items()
        if not (0 <= row + delta_row < env.height and 0 <= col + delta_col < env.width)
    )

    observations: list[int] = []
    truncations: list[bool] = []
    for _ in range(env._max_steps + WALL_BUMPS_BEYOND_BUDGET):
        observation, _, terminated, truncated, _ = env.step(illegal_action)
        observations.append(_obs_hash(observation))
        truncations.append(truncated)
        assert not terminated, "An illegal move must not terminate the episode."

    logger.info(
        f"{len(observations)} illegal moves from {(row, col)} (budget {env._max_steps}): "
        f"{len(set(observations))} distinct observation(s), "
        f"step counter {env._current_step}, truncation reported: {any(truncations)}"
    )
    assert (
        len(set(observations)) == 1
    ), "Illegal moves should leave the observation unchanged."
    assert env._agent_location == (
        row,
        col,
    ), "Agent must not move on an illegal action."
    assert not any(truncations), (
        "env v1 returns truncated=False on the illegal-move branch, so an agent that "
        "only bumps walls runs forever."
    )
