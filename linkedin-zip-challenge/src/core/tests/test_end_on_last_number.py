# src/core/tests/test_end_on_last_number.py
"""Every judge in the project agrees: a solution ends on the highest number.

Until 2026-09-19 all of them accepted a walk that collected the last number and kept going
(the env, `dfs.py`, A*, CP-SAT, the fitness score, `verify.is_solution` and the VLM
scorer). LinkedIn's help page leaves the end cell open, but the generator always ends the
path on the last number and rule write-ups say it must. The judges are separate on purpose
-- `verify.py` exists so the heuristics are not graded by the score they optimise -- so the
rule is pinned here once, against all of them.

On this board depth-first search, which tries "right" first, reaches the loose-only walk
before the real solution, so the board separates the two readings for the solvers too.
"""

import pytest

from src.core.rl.action_space import path_to_actions
from src.core.rl.baselines import make_eval_env
from src.core.rl.rl_env_v2 import PuzzleSample
from src.core.solvers import a_star, cp, dfs
from src.core.solvers.verify import is_solution
from src.core.utils import Puzzle, calculate_fitness_score, parse_puzzle_layout
from src.core.vl_models.score_predictions import path_is_legal

STRICT_SOLUTION = [(0, 0), (1, 0), (1, 1), (0, 1)]
# Covers every cell and collects 1 then 2, but walks on past the 2.
PAST_THE_LAST_NUMBER = [(0, 0), (0, 1), (1, 1), (1, 0)]


def _board() -> Puzzle:
    puzzle = parse_puzzle_layout([["01", "02"], ["  ", "  "]])
    puzzle["walls"] = set()
    return puzzle


@pytest.mark.parametrize(
    "solve",
    [
        dfs.solve_puzzle,
        a_star.solve_puzzle_a_star,
        a_star.solve_puzzle_a_star_sortedlist,
        cp.solve_puzzle_cp,
    ],
)
def test_exact_solvers_return_the_path_that_ends_on_the_last_number(solve) -> None:
    assert solve(_board()) == STRICT_SOLUTION


def test_verify_rejects_a_walk_past_the_last_number() -> None:
    assert is_solution(_board(), STRICT_SOLUTION)
    assert not is_solution(_board(), PAST_THE_LAST_NUMBER)


def test_fitness_score_withholds_the_jackpot_past_the_last_number() -> None:
    score, perfect = calculate_fitness_score(_board(), STRICT_SOLUTION)
    past, _ = calculate_fitness_score(_board(), PAST_THE_LAST_NUMBER)

    assert score == perfect
    assert past < perfect


def test_vlm_scorer_rejects_a_walk_past_the_last_number() -> None:
    assert path_is_legal(_board(), STRICT_SOLUTION)
    assert not path_is_legal(_board(), PAST_THE_LAST_NUMBER)


@pytest.mark.parametrize(
    ("walk", "solved"), [(STRICT_SOLUTION, True), (PAST_THE_LAST_NUMBER, False)]
)
def test_rl_env_scores_only_the_walk_that_ends_on_the_last_number(
    walk: list[tuple[int, int]], solved: bool
) -> None:
    env = make_eval_env(PuzzleSample(puzzle=_board(), solution_path=STRICT_SOLUTION))
    env.reset(seed=0)
    for action in path_to_actions(walk):
        _, _, terminated, _, info = env.step(action)

    assert terminated
    assert info["solved"] is solved
