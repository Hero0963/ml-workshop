# src/core/rl/diagnose_env_v1.py
"""A0 diagnostics for the legacy `PuzzleEnv` (src/core/rl/rl_env.py).

Run:  uv run python -m src.core.rl.diagnose_env_v1

Produces the experimental evidence the 2026-08-15 RL restart plan asks for:

* Probe 1 -- can a ground-truth solution terminate the episode at all?
* Probe 2 -- why not: when does `_next_waypoint_idx` actually advance?
* Probe 3 -- is the observation sequence a 2-cycle while oscillating?
* Probe 4 -- does a deterministic policy built from that 2-cycle ever escape?
* Probe 5 -- illegal moves: a degenerate 1-cycle that never truncates.

The environment is only inspected, never modified: v1 stays as the control for v2.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

from src.core.puzzle_generation.puzzle_generator import generate_puzzle
from src.core.rl.action_space import ACTION_DELTAS, path_to_actions
from src.core.rl.rl_env import PuzzleEnv
from src.core.tests.conftest import puzzles_to_test
from src.core.utils import Puzzle

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "logs" / "rl_diagnostics"
GENERATOR_SEED = 42
GENERATED_GRID = (4, 4)
OSCILLATION_ROUND_TRIPS = 4
# Bump past the step budget so "the episode never ends" is measured, not assumed.
WALL_BUMPS_BEYOND_BUDGET = 10
DETERMINISTIC_POLICY_STEP_CAP = 500


def _obs_hash(observation: np.ndarray) -> str:
    return f"{hash(observation.tobytes()) & 0xFFFFFFFF:08x}"


def _probe_ground_truth_replay(
    puzzle: Puzzle, solution_path: list[tuple[int, int]], puzzle_id: str
) -> dict[str, object]:
    """Feeds the ground-truth path in step by step and reports the final state."""
    env = PuzzleEnv(puzzle)
    env.reset()

    terminated = truncated = False
    total_reward = 0.0
    waypoint_advances = 0
    for action in path_to_actions(solution_path):
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if env._next_waypoint_idx > waypoint_advances:
            waypoint_advances = env._next_waypoint_idx
        if terminated or truncated:
            break

    outcome = {
        "puzzle_id": puzzle_id,
        "grid_size": list(env.grid_size),
        "path_length": len(solution_path),
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": round(total_reward, 3),
        "waypoints_reached": env._next_waypoint_idx,
        "waypoint_count": len(env.waypoints),
        "cells_visited": len(set(env.path_taken)),
        "visitable_cells": env.total_visitable_cells,
    }
    logger.info(
        f"[probe 1] {puzzle_id}: terminated={terminated} "
        f"waypoints={outcome['waypoints_reached']}/{outcome['waypoint_count']} "
        f"cells={outcome['cells_visited']}/{outcome['visitable_cells']} "
        f"total_reward={outcome['total_reward']}"
    )
    return outcome


def _probe_waypoint_advance(
    puzzle: Puzzle, solution_path: list[tuple[int, int]]
) -> dict[str, object]:
    """Shows that waypoint 1 is only credited by *re-entering* the start cell."""
    env = PuzzleEnv(puzzle)
    env.reset()

    start_pos = env.start_pos
    idx_after_reset = env._next_waypoint_idx
    target_after_reset = env.waypoints[idx_after_reset]

    leave_action = path_to_actions(solution_path[:2])[0]
    env.step(leave_action)
    idx_after_leaving = env._next_waypoint_idx

    return_action = path_to_actions([solution_path[1], solution_path[0]])[0]
    _, return_reward, _, _, _ = env.step(return_action)
    idx_after_returning = env._next_waypoint_idx

    outcome = {
        "start_pos": list(start_pos),
        "target_after_reset": list(target_after_reset),
        "start_is_first_target": tuple(start_pos) == tuple(target_after_reset),
        "idx_after_reset": idx_after_reset,
        "idx_after_leaving_start": idx_after_leaving,
        "idx_after_stepping_back_onto_start": idx_after_returning,
        "reward_for_stepping_back": round(return_reward, 3),
    }
    logger.info(
        f"[probe 2] reset target == start cell: {outcome['start_is_first_target']}; "
        f"idx reset={idx_after_reset} -> leave={idx_after_leaving} -> "
        f"step back onto start={idx_after_returning} "
        f"(reward {outcome['reward_for_stepping_back']})"
    )
    return outcome


def _probe_detour_replay(
    puzzle: Puzzle, solution_path: list[tuple[int, int]]
) -> dict[str, object]:
    """Replays the ground truth prefixed with a step off and back onto the start cell.

    That detour is an illegal Zip solution (the start cell is used twice), but it is
    the only way the waypoint chain can be completed -- so it should terminate where
    the legal path cannot.
    """
    detour_path = [solution_path[0], solution_path[1], *solution_path]
    env = PuzzleEnv(puzzle)
    env.reset()

    terminated = truncated = False
    total_reward = 0.0
    final_reward = 0.0
    for action in path_to_actions(detour_path):
        _, final_reward, terminated, truncated, _ = env.step(action)
        total_reward += final_reward
        if terminated or truncated:
            break

    outcome = {
        "detour_length": len(detour_path),
        "legal_one_stroke": len(set(detour_path)) == len(detour_path),
        "terminated": terminated,
        "truncated": truncated,
        "final_reward": round(final_reward, 3),
        "total_reward": round(total_reward, 3),
        "waypoints_reached": env._next_waypoint_idx,
        "waypoint_count": len(env.waypoints),
    }
    logger.info(
        f"[probe 6] ground truth + start-cell detour: terminated={terminated} "
        f"final_reward={outcome['final_reward']} total_reward={outcome['total_reward']} "
        f"waypoints={outcome['waypoints_reached']}/{outcome['waypoint_count']} "
        f"legal_one_stroke={outcome['legal_one_stroke']}"
    )
    return outcome


def _probe_observation_cycle(
    puzzle: Puzzle, solution_path: list[tuple[int, int]]
) -> dict[str, object]:
    """Oscillates between two visited cells and hashes every observation."""
    env = PuzzleEnv(puzzle)
    env.reset()
    for action in path_to_actions(solution_path[:3]):
        env.step(action)

    cell_a, cell_b = solution_path[2], solution_path[1]
    action_to_a = path_to_actions([cell_b, cell_a])[0]
    action_to_b = path_to_actions([cell_a, cell_b])[0]

    trace: list[dict[str, object]] = []
    for _ in range(OSCILLATION_ROUND_TRIPS):
        for action in (action_to_b, action_to_a):
            observation, reward, terminated, truncated, _ = env.step(action)
            trace.append(
                {
                    "action": action,
                    "agent": list(env._agent_location),
                    "obs_hash": _obs_hash(observation),
                    "reward": round(reward, 3),
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )

    distinct = sorted({entry["obs_hash"] for entry in trace})
    outcome = {
        "cell_a": list(cell_a),
        "cell_b": list(cell_b),
        "steps": len(trace),
        "distinct_observations": len(distinct),
        "observation_hashes": distinct,
        "trace": trace,
    }
    logger.info(
        f"[probe 3] oscillating {cell_a} <-> {cell_b} for {len(trace)} steps produced "
        f"{len(distinct)} distinct observations: {distinct}"
    )
    return outcome


def _probe_deterministic_policy_lock_in(
    puzzle: Puzzle, solution_path: list[tuple[int, int]]
) -> dict[str, object]:
    """Runs the deterministic policy induced by the 2-cycle until the env gives up."""
    env = PuzzleEnv(puzzle)
    observation, _ = env.reset()
    for action in path_to_actions(solution_path[:3]):
        observation, _, _, _, _ = env.step(action)

    cell_a, cell_b = solution_path[2], solution_path[1]
    policy: dict[str, int] = {
        _obs_hash(observation): path_to_actions([cell_a, cell_b])[0],
    }
    observation, _, _, _, _ = env.step(policy[_obs_hash(observation)])
    policy[_obs_hash(observation)] = path_to_actions([cell_b, cell_a])[0]

    visited_cells: set[tuple[int, int]] = set()
    steps = 0
    escaped = False
    truncated = False
    while steps < DETERMINISTIC_POLICY_STEP_CAP:
        obs_key = _obs_hash(observation)
        if obs_key not in policy:
            escaped = True
            break
        observation, _, terminated, truncated, _ = env.step(policy[obs_key])
        visited_cells.add(env._agent_location)
        steps += 1
        if terminated or truncated:
            break

    outcome = {
        "policy_states": len(policy),
        "steps_until_episode_end": steps,
        "escaped_the_cycle": escaped,
        "truncated": truncated,
        "cells_visited_during_lock_in": sorted(list(cell) for cell in visited_cells),
        "env_max_steps": env._max_steps,
    }
    logger.info(
        f"[probe 4] a 2-state deterministic policy ran {steps} steps, "
        f"escaped={escaped}, truncated={truncated}, "
        f"cells touched={outcome['cells_visited_during_lock_in']}"
    )
    return outcome


def _probe_illegal_move_loop(puzzle: Puzzle) -> dict[str, object]:
    """Bumps a boundary repeatedly: observation frozen, episode never truncates."""
    env = PuzzleEnv(puzzle)
    env.reset()
    row, col = env._agent_location
    illegal_action = next(
        action
        for action, (delta_row, delta_col) in ACTION_DELTAS.items()
        if not (0 <= row + delta_row < env.height and 0 <= col + delta_col < env.width)
    )

    bump_count = env._max_steps + WALL_BUMPS_BEYOND_BUDGET
    hashes: list[str] = []
    rewards: list[float] = []
    truncations: list[bool] = []
    for _ in range(bump_count):
        observation, reward, _, truncated, _ = env.step(illegal_action)
        hashes.append(_obs_hash(observation))
        rewards.append(round(reward, 3))
        truncations.append(truncated)

    outcome = {
        "start_pos": [row, col],
        "illegal_action": illegal_action,
        "bump_count": bump_count,
        "distinct_observations": len(set(hashes)),
        "distinct_rewards": sorted(set(rewards)),
        "agent_moved": env._agent_location != (row, col),
        "step_counter": env._current_step,
        "env_max_steps": env._max_steps,
        "any_truncation_reported": any(truncations),
    }
    logger.info(
        f"[probe 5] {bump_count} illegal moves from {[row, col]} "
        f"(budget {env._max_steps}): {outcome['distinct_observations']} distinct "
        f"observation(s), agent_moved={outcome['agent_moved']}, "
        f"truncation reported={outcome['any_truncation_reported']}"
    )
    return outcome


def run_diagnostics() -> dict[str, object]:
    """Runs every probe and returns the collected evidence."""
    reference_puzzle, reference_solution = puzzles_to_test[1][0], puzzles_to_test[1][1]

    replays = [
        _probe_ground_truth_replay(puzzle, solution, puzzle_id)
        for puzzle, solution, puzzle_id in puzzles_to_test
    ]

    random.seed(GENERATOR_SEED)
    generated = generate_puzzle(
        m=GENERATED_GRID[0], n=GENERATED_GRID[1], has_walls=False, num_blocked_cells=0
    )
    if generated is not None:
        generated_puzzle, generated_solution = generated
        replays.append(
            _probe_ground_truth_replay(
                generated_puzzle,
                generated_solution,
                f"generated_{GENERATED_GRID[0]}x{GENERATED_GRID[1]}_seed{GENERATOR_SEED}",
            )
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "src/core/rl/rl_env.py (v1, unmodified)",
        "generator_seed": GENERATOR_SEED,
        "probe_1_ground_truth_replay": replays,
        "probe_2_waypoint_advance": _probe_waypoint_advance(
            reference_puzzle, reference_solution
        ),
        "probe_6_detour_replay": [
            _probe_detour_replay(puzzle, solution)
            for puzzle, solution, _ in puzzles_to_test
        ],
        "probe_3_observation_cycle": _probe_observation_cycle(
            reference_puzzle, reference_solution
        ),
        "probe_4_deterministic_policy_lock_in": _probe_deterministic_policy_lock_in(
            reference_puzzle, reference_solution
        ),
        "probe_5_illegal_move_loop": _probe_illegal_move_loop(reference_puzzle),
    }


def main() -> None:
    evidence = run_diagnostics()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "env_v1_diagnosis.json"
    artifact_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    solvable = [r for r in evidence["probe_1_ground_truth_replay"] if r["terminated"]]
    logger.info(
        f"Ground-truth replays that terminated successfully: "
        f"{len(solvable)}/{len(evidence['probe_1_ground_truth_replay'])}"
    )
    logger.info(f"Evidence written to {artifact_path}")


if __name__ == "__main__":
    main()
