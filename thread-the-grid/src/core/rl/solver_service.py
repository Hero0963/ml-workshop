# src/core/rl/solver_service.py
"""Runs a trained policy as a solver: a puzzle goes in, a path comes out.

This is the serving side of the RL track (stage A5). Everything else in `src/core/rl/`
exists to *produce* a checkpoint; this module is the only place that *consumes* one
outside an experiment script, and it is deliberately thin: it reuses the same env, the
same action masking and the same rollout loop the evaluation uses, so a number measured
in a report is the number this endpoint delivers.

Three things it has to get right, because each of them would otherwise surface as a 500:

*   **The checkpoint may not be there.** `models/` is not in version control, so a fresh
    clone or a container without the volume mounted has no policy at all. That is a
    service-not-ready condition, not a bug: `ModelUnavailableError` -> 503.
*   **The board may have no model.** The track trained specific sizes; anything else is a
    request the service cannot answer, not a failure: `UnsupportedBoardError` -> 400.
*   **A solver is allowed to fail.** Returning `None` means "no solution found within the
    attempt budget", which is an ordinary 200 answer with an explanation, exactly like the
    heuristic solvers already do.

⚠ The policy is *not* the best solver here. CP-SAT is exact and faster; this endpoint
exists so the learning track has a visible endpoint, and because a solver whose quality
depends on how much inference budget you give it is a different shape of artifact worth
having in the API. See `ai-collab/notes/inference-and-serving.md`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.core.rl.baselines import make_eval_env
from src.core.rl.rl_env_v2 import PuzzleSample
from src.core.utils import Puzzle

MODEL_ROOT = Path(__file__).resolve().parents[3] / "models" / "rl_a2"
CHECKPOINT_DIRNAME = "checkpoints"
FINAL_CHECKPOINT_NAME = "model_final"

# Which checkpoint answers for which board. One model for all three, because the
# multi-size arm won on every board against a size-specific control trained on the same
# data and cost slightly less to train than the three controls together. It is also the
# only policy that can answer a 5x5 at all -- the older packs held sizes 4 and 6 only.
# The per-size shape is kept because nothing guarantees the next model is universal.
#
# `_e6` is the same recipe stopped at 6 epochs instead of 10 (2026-09-12). Fewer epochs
# leave the policy less peaked, which costs single-attempt accuracy and buys diversity --
# and this endpoint spends `DEFAULT_ATTEMPTS` rollouts, so diversity is what it wants:
# 6x6 best-of-32 0.8535 -> 0.9465 and 7.76 -> 5.24 attempts per puzzle, 4x4 0.9917 ->
# 0.9953. Deterministic does not regress either (4x4 0.9410, 5x5 0.7701, 6x6 0.5430).
# ⚠ The trade is real at the other end: best-of-1 drops (6x6 0.4810 -> 0.4265), so a
# caller that passes `attempts=1` is better served by the 10-epoch `bc_multi_456`.
RUN_ID_BY_SIZE: dict[int, str] = {
    4: "bc_multi_456_e6",
    5: "bc_multi_456_e6",
    6: "bc_multi_456_e6",
}

# The first waypoint is where the pen starts, by the rules of the puzzle.
FIRST_WAYPOINT_NUMBER = 1

# Best-of-N budget. 32 is the number the 6x6 result is quoted at (0.9465 on held-out
# test, 5.24 attempts per puzzle); 4x4 clears its bar at 2 (0.9529) and simply stops
# early. Attempts halt at the first solve, so this is a ceiling, not a cost.
DEFAULT_ATTEMPTS = 32


class RLSolverError(Exception):
    """Base for the two conditions a caller has to be able to tell apart."""


class ModelUnavailableError(RLSolverError):
    """The checkpoint this board needs is not on disk."""


class UnsupportedBoardError(RLSolverError):
    """No policy was ever trained for this board size."""


def supported_sizes() -> list[int]:
    return sorted(RUN_ID_BY_SIZE)


def checkpoint_path(run_id: str) -> Path:
    return MODEL_ROOT / run_id / CHECKPOINT_DIRNAME / f"{FINAL_CHECKPOINT_NAME}.zip"


@lru_cache(maxsize=4)
def load_policy(run_id: str) -> Any:
    """Loads a checkpoint once and keeps it.

    Imports of torch and sb3 are deferred to here on purpose: the API imports this module
    at startup to register the solver, and paying two seconds of CUDA initialisation for
    an endpoint nobody may call makes every container start slower.
    """
    path = checkpoint_path(run_id)
    if not path.exists():
        raise ModelUnavailableError(
            f"No checkpoint at {path}. `models/` is not in version control -- train the "
            f"goal or mount the model volume."
        )

    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    logger.info(f"Loading RL policy {run_id} from {path}")
    return MaskablePPO.load(path, device="cpu")


def inference_sample(puzzle: Puzzle) -> PuzzleSample:
    """Wraps a puzzle as the env expects, without knowing its solution.

    `PuzzleSample` carries `solution_path` because it was built for training, where the
    path is both the label and the reverse curriculum's ruler. With
    `reverse_curriculum_k=None` the env reads exactly one element of it -- the cell the
    pen starts on -- and that is derivable from the puzzle itself: waypoint 1. Handing it
    a one-element path is therefore not a stub standing in for missing data, it is all
    the data the env will read. `test_solver_service.py` pins that invariant, so this
    breaks loudly if the env ever starts reading further into the path.
    """
    num_map = puzzle["num_map"]
    if FIRST_WAYPOINT_NUMBER not in num_map:
        raise UnsupportedBoardError(
            f"The puzzle has no waypoint {FIRST_WAYPOINT_NUMBER}; there is nowhere to start."
        )
    return PuzzleSample(puzzle=puzzle, solution_path=[num_map[FIRST_WAYPOINT_NUMBER]])


def _rollout(policy: Any, env: Any, rng: np.random.Generator, deterministic: bool):
    """Plays one episode and returns the walked path, or None if it dead-ends."""
    _, info = env.reset(seed=int(rng.integers(2**31 - 1)))
    # `info["agent_location"]` rather than the env's private attribute: the info dict is
    # the env's published interface and already carries every field this needs.
    path = [info["agent_location"]]
    while True:
        mask = env.action_masks()
        if not mask.any():
            return None
        action, _ = policy.predict(
            env.observation(), deterministic=deterministic, action_masks=mask
        )
        _, _, terminated, truncated, info = env.step(int(action))
        path.append(info["agent_location"])
        if terminated or truncated:
            return path if info["solved"] else None


def solve_puzzle_rl(
    puzzle: Puzzle, attempts: int = DEFAULT_ATTEMPTS, seed: int = 20260815
) -> list[tuple[int, int]] | None:
    """Solves with a trained policy, spending up to `attempts` rollouts.

    The first rollout is the deterministic one (argmax at every step); the rest are
    sampled from the same masked distribution. Stopping at the first success is what
    makes the budget a ceiling rather than a cost, and it is legitimate rather than
    guessing because a Zip solution verifies itself -- the env only reports `solved`
    when every cell is covered and every waypoint was collected in order.
    """
    size = puzzle["grid_size"][0]
    run_id = RUN_ID_BY_SIZE.get(size)
    if run_id is None:
        raise UnsupportedBoardError(
            f"No RL policy for a {size}x{size} board; trained sizes are {supported_sizes()}."
        )

    policy = load_policy(run_id)
    sample = inference_sample(puzzle)
    env = make_eval_env(sample)
    rng = np.random.default_rng(seed)

    for attempt in range(max(1, attempts)):
        path = _rollout(policy, env, rng, deterministic=(attempt == 0))
        if path is not None:
            logger.info(
                f"RL solver ({run_id}) solved a {size}x{size} on attempt {attempt + 1}"
            )
            return path

    logger.info(f"RL solver ({run_id}) found no solution in {attempts} attempts")
    return None
