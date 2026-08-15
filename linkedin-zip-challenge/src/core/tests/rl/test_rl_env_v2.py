# src/core/tests/rl/test_rl_env_v2.py
"""A1 unit tests for the one-stroke environment `PuzzleEnvV2`.

The first test is the one v1 could never pass: replaying a ground-truth solution has to
finish with the success reward. The rest pin the masking rules, the dead-end boundary
(`MaskablePPO` breaks if an all-False mask reaches the sampler) and the reward edges.
"""

import random

import numpy as np
import pytest
from loguru import logger

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.action_space import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    path_to_actions,
)
from src.core.rl.rl_env_v2 import (
    CH_AGENT,
    CH_VISITED,
    CH_WP_DONE,
    GRID_PAD,
    NUM_ACTIONS,
    PuzzleEnvV2,
    PuzzleSample,
    SUCCESS_REWARD,
)
from src.core.tests.conftest import puzzles_to_test
from src.core.utils import Puzzle, parse_puzzle_layout

GENERATOR_SEED = 42
REVERSE_CURRICULUM_K = 3


def _make_env(
    puzzle: Puzzle, solution_path: list[tuple[int, int]], **kwargs
) -> PuzzleEnvV2:
    return PuzzleEnvV2([PuzzleSample(puzzle, solution_path)], **kwargs)


def _replay(
    env: PuzzleEnvV2, actions: list[int]
) -> tuple[float, bool, bool, dict[str, object]]:
    total_reward = 0.0
    terminated = truncated = False
    info: dict[str, object] = {}
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, terminated, truncated, info


@pytest.mark.parametrize(
    "puzzle_data, expected_solution, puzzle_id",
    puzzles_to_test,
    ids=[p[2] for p in puzzles_to_test],
)
def test_ground_truth_replay_solves_fixture_puzzles(
    puzzle_data: Puzzle, expected_solution: list[tuple[int, int]], puzzle_id: str
) -> None:
    """The check env v1 failed 0/7: a legal one-stroke solution must terminate."""
    env = _make_env(puzzle_data, expected_solution, shaping_lambda=0.0)
    env.reset()

    total_reward, terminated, truncated, info = _replay(
        env, path_to_actions(expected_solution)
    )
    logger.info(
        f"{puzzle_id}: reward={total_reward} terminated={terminated} info={info}"
    )

    assert terminated, f"{puzzle_id}: the ground-truth solution did not terminate."
    assert not truncated, f"{puzzle_id}: episode was truncated."
    assert info["solved"], f"{puzzle_id}: terminated without being solved."
    assert total_reward == pytest.approx(SUCCESS_REWARD), (
        f"{puzzle_id}: with shaping off a solved episode must score exactly "
        f"{SUCCESS_REWARD}, got {total_reward}."
    )


@pytest.mark.parametrize(
    "rows, cols, has_walls",
    [(4, 4, False), (5, 5, True), (6, 6, True)],
    ids=["generated_4x4_open", "generated_5x5_walls", "generated_6x6_walls"],
)
def test_ground_truth_replay_solves_generated_puzzles(
    rows: int, cols: int, has_walls: bool
) -> None:
    random.seed(GENERATOR_SEED)
    result = generate_puzzle(m=rows, n=cols, has_walls=has_walls, num_blocked_cells=0)
    assert result is not None, "Puzzle generation failed; cannot run the replay check."

    puzzle, solution_path = result
    env = _make_env(puzzle, solution_path, shaping_lambda=0.0)
    env.reset()

    total_reward, terminated, _, info = _replay(env, path_to_actions(solution_path))
    assert terminated and info["solved"], f"{rows}x{cols}: generated puzzle not solved."
    assert total_reward == pytest.approx(SUCCESS_REWARD)


def test_observations_stay_inside_the_declared_space() -> None:
    puzzle, solution, _ = puzzles_to_test[1]
    env = _make_env(puzzle, solution)
    observation, _ = env.reset()

    assert env.observation_space.contains(
        observation
    ), "reset() left the observation space."
    assert observation["grid"].shape == (8, GRID_PAD, GRID_PAD)

    for action in path_to_actions(solution)[:5]:
        observation, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(
            observation
        ), "step() left the observation space."


def test_visited_cells_and_the_start_waypoint_are_encoded() -> None:
    puzzle, solution, _ = puzzles_to_test[1]
    env = _make_env(puzzle, solution)
    observation, _ = env.reset()

    start = solution[0]
    assert observation["grid"][CH_VISITED][start] == 1.0
    assert observation["grid"][CH_AGENT][start] == 1.0
    assert (
        observation["grid"][CH_WP_DONE][start] == 1.0
    ), "Standing on number 1 collects it at reset (dfs.py:72-77); v1 got this wrong."


def test_visited_cells_are_masked_out() -> None:
    """A one-stroke walk can never step back, which is why no 2-cycle can form."""
    puzzle, solution, _ = puzzles_to_test[1]
    env = _make_env(puzzle, solution)
    env.reset()

    forward = path_to_actions(solution[:2])[0]
    env.step(forward)
    backward = path_to_actions(solution[1::-1])[0]

    assert not env.action_masks()[
        backward
    ], "Stepping back onto a visited cell must be masked."


def _two_by_two_puzzle() -> Puzzle:
    """`01` and `03` are neighbours, so `03` must be masked until `02` is collected."""
    puzzle = parse_puzzle_layout([["01", "03"], ["  ", "02"]])
    puzzle["walls"] = set()
    return puzzle


def test_out_of_order_waypoints_are_masked() -> None:
    puzzle = _two_by_two_puzzle()
    solution = [(0, 0), (1, 0), (1, 1), (0, 1)]
    env = _make_env(puzzle, solution)
    env.reset()

    masks = env.action_masks()
    assert not masks[ACTION_RIGHT], "Number 3 must be masked while number 2 is pending."
    assert masks[ACTION_DOWN], "The empty cell below must stay legal."

    total_reward, terminated, _, info = _replay(env, path_to_actions(solution))
    assert (
        terminated and info["solved"]
    ), "The in-order path must still solve the puzzle."


def test_walls_and_blocked_cells_are_masked() -> None:
    puzzle = _two_by_two_puzzle()
    puzzle["walls"] = {tuple(sorted(((0, 0), (1, 0))))}
    env = _make_env(puzzle, [(0, 0), (1, 0), (1, 1), (0, 1)])
    env.reset()

    assert not env.action_masks()[ACTION_DOWN], "A wall must mask the move through it."

    blocked = parse_puzzle_layout([["01", "03"], ["xx", "02"]])
    blocked["walls"] = set()
    blocked_env = _make_env(blocked, [(0, 0), (0, 1), (1, 1)])
    blocked_env.reset()
    assert not blocked_env.action_masks()[ACTION_DOWN], "A blocked cell must be masked."


def _dead_end_puzzle() -> tuple[Puzzle, list[tuple[int, int]]]:
    """2x3 board where going right first traps the agent in the bottom-left corner."""
    puzzle = parse_puzzle_layout([["01", "  ", "  "], ["  ", "  ", "02"]])
    puzzle["walls"] = set()
    solution = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    return puzzle, solution


def test_dead_end_terminates_before_an_all_false_mask_is_sampled() -> None:
    """The boundary `MaskablePPO` cannot survive: every action masked."""
    puzzle, solution = _dead_end_puzzle()
    env = _make_env(puzzle, solution, shaping_lambda=0.0)
    env.reset()

    total_reward, terminated, truncated, info = _replay(
        env, [ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
    )
    logger.info(f"dead-end episode: reward={total_reward} info={info}")

    assert terminated, "A dead end must terminate the episode."
    assert not truncated
    assert info["dead_end"] and not info["solved"]
    assert total_reward == pytest.approx(
        0.0
    ), "A dead end must not pay the success bonus."
    assert (
        not env.action_masks().any()
    ), "The episode ended exactly at the all-masked state."


def test_ground_truth_still_solves_the_dead_end_board() -> None:
    """The same board is solvable, so the dead end is the agent's fault, not the env's."""
    puzzle, solution = _dead_end_puzzle()
    env = _make_env(puzzle, solution, shaping_lambda=0.0)
    env.reset()

    total_reward, terminated, _, info = _replay(env, path_to_actions(solution))
    assert terminated and info["solved"]
    assert total_reward == pytest.approx(SUCCESS_REWARD)


def test_reverse_curriculum_starts_k_cells_from_the_end() -> None:
    puzzle, solution, _ = puzzles_to_test[1]
    env = _make_env(
        puzzle, solution, reverse_curriculum_k=REVERSE_CURRICULUM_K, shaping_lambda=0.0
    )
    _, info = env.reset()

    assert info["agent_location"] == solution[-REVERSE_CURRICULUM_K]
    assert info["coverage"] == pytest.approx(
        (len(solution) - REVERSE_CURRICULUM_K + 1) / len(solution)
    )

    remaining = path_to_actions(solution[-REVERSE_CURRICULUM_K:])
    assert len(remaining) == REVERSE_CURRICULUM_K - 1

    total_reward, terminated, _, info = _replay(env, remaining)
    assert (
        terminated and info["solved"]
    ), "The tail of the solution must finish the episode."
    assert total_reward == pytest.approx(SUCCESS_REWARD)


def test_reverse_curriculum_rejects_an_already_solved_start() -> None:
    puzzle, solution, _ = puzzles_to_test[1]
    with pytest.raises(ValueError, match="reverse_curriculum_k"):
        _make_env(puzzle, solution, reverse_curriculum_k=1)


def test_invalid_action_ends_the_episode_without_reward() -> None:
    puzzle, solution, _ = puzzles_to_test[1]
    env = _make_env(puzzle, solution)
    env.reset()

    masks = env.action_masks()
    illegal_action = int(np.argmin(masks))
    assert not masks[
        illegal_action
    ], "This board should offer at least one illegal move."

    _, reward, terminated, truncated, info = env.step(illegal_action)
    assert terminated and not truncated
    assert reward == pytest.approx(0.0)
    assert info["invalid_action"]


def test_shaping_pays_out_only_on_progress() -> None:
    """With shaping on, a solved episode scores the bonus plus a bounded shaping sum."""
    puzzle, solution, _ = puzzles_to_test[1]
    shaping_lambda = 0.2
    env = _make_env(puzzle, solution, shaping_lambda=shaping_lambda)
    env.reset()

    total_reward, terminated, _, _ = _replay(env, path_to_actions(solution))
    assert terminated
    assert SUCCESS_REWARD < total_reward < SUCCESS_REWARD + shaping_lambda, (
        f"Coverage shaping over a full episode must stay below lambda={shaping_lambda}, "
        f"got {total_reward - SUCCESS_REWARD}."
    )


def test_masks_never_allow_an_illegal_move_during_a_random_rollout() -> None:
    """Fuzz: follow only legal actions and assert the walk stays a valid one-stroke path."""
    random.seed(GENERATOR_SEED)
    result = generate_puzzle(m=5, n=5, has_walls=True, num_blocked_cells=0)
    assert result is not None

    puzzle, solution = result
    env = _make_env(puzzle, solution)
    env.reset()
    rng = np.random.default_rng(GENERATOR_SEED)

    walked = [env._agent_location]
    for _ in range(len(solution)):
        masks = env.action_masks()
        if not masks.any():
            break
        action = int(rng.choice(np.flatnonzero(masks)))
        _, _, terminated, truncated, info = env.step(action)
        walked.append(info["agent_location"])
        if terminated or truncated:
            break

    assert len(walked) == len(
        set(walked)
    ), "A masked rollout must never revisit a cell."
    assert len(walked) <= len(
        solution
    ), "A one-stroke walk cannot exceed the cell count."
    assert all(
        cell not in puzzle["blocked_cells"] for cell in walked
    ), "A masked rollout must never enter a blocked cell."
    assert len(env.action_masks()) == NUM_ACTIONS
