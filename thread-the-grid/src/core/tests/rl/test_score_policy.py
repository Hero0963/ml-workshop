# src/core/tests/rl/test_score_policy.py
"""Pins the best-of-N accounting every published RL number was measured with.

No checkpoint is needed: the accounting is a property of the scorer, not of a model, so
it is exercised with scripted policies on hand-made boards and with the baselines.
"""

import json
import pickle
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from src.core.rl.action_space import ACTION_RIGHT, path_to_actions
from src.core.rl.baselines import POLICIES, PolicyFn, make_eval_env, run_episode
from src.core.rl.eval_set import MANIFEST_FILENAME, export_split
from src.core.rl.generate_dataset_v2 import split_digest
from src.core.rl.rl_env_v2 import PuzzleEnvV2, PuzzleSample
from src.core.rl.score_policy import (
    attempts_until_solved,
    best_of_n,
    build_arg_parser,
    score,
    score_samples,
    table,
)
from src.core.utils import parse_puzzle_layout

SEED = 20260815

# 2x3 board from `test_rl_env_v2`: going right first strands the bottom-left corner, so a
# policy that opens with RIGHT can never solve it.
LAYOUT = [["01", "  ", "  "], ["  ", "  ", "02"]]
SOLUTION = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]


def _board() -> PuzzleSample:
    puzzle = parse_puzzle_layout(LAYOUT)
    puzzle["walls"] = set()
    return PuzzleSample(puzzle, SOLUTION)


def _expert() -> PolicyFn:
    """Replays the solution. Every episode it plays is complete, so a cycle stays in step."""
    actions = path_to_actions(SOLUTION)
    steps = count()

    def policy(env: PuzzleEnvV2, rng: np.random.Generator) -> int:
        return actions[next(steps) % len(actions)]

    return policy


def _opens_right(env: PuzzleEnvV2, rng: np.random.Generator) -> int:
    mask = env.action_masks()
    return ACTION_RIGHT if mask[ACTION_RIGHT] else int(np.flatnonzero(mask)[0])


def test_the_first_success_is_counted_and_nothing_runs_after_it() -> None:
    outcomes = iter([False, False, True, True])
    assert attempts_until_solved(lambda: next(outcomes), max_attempts=10) == 3
    assert next(outcomes) is True, "An attempt was played after the solve."


def test_an_unsolved_puzzle_spends_the_whole_budget() -> None:
    played = []
    assert attempts_until_solved(lambda: played.append(1) or False, 4) is None
    assert len(played) == 4


def test_a_zero_budget_plays_nothing() -> None:
    assert attempts_until_solved(lambda: pytest.fail("played"), 0) is None


def test_best_of_n_reads_every_n_off_one_pass() -> None:
    table_ = best_of_n([1, 3, None, 2], max_attempts=4, reported_n=(1, 2, 4, 8))

    assert table_ == {
        "1": {"solve_rate": 0.25, "episodes_per_puzzle": 1.0},
        "2": {"solve_rate": 0.5, "episodes_per_puzzle": 1.75},
        "4": {"solve_rate": 0.75, "episodes_per_puzzle": 2.5},
    }, "N above the budget must be omitted; cost is min(first success, N), else N."


def test_opening_right_never_solves_under_the_strict_judge() -> None:
    """The precondition the next test leans on, and a pin on the judge the scorer inherits.

    Opening right still covers every cell with the numbers in order, but ends on (1, 0)
    rather than on 02: a solve under the loose reading, not under the strict one.
    """
    env = make_eval_env(_board())
    result = run_episode(env, _opens_right, np.random.default_rng(SEED))
    assert result["coverage"] == 1.0
    assert not result["solved"]


def test_the_deterministic_episode_is_not_attempt_one() -> None:
    solved, first_success = score_samples(
        [_board()], _expert(), _opens_right, seed=SEED, max_attempts=3
    )
    assert solved == 1
    assert first_success == [None], "A deterministic solve must not count as attempt 1."


def test_an_expert_solves_on_the_first_sampled_attempt() -> None:
    solved, first_success = score_samples(
        [_board(), _board()], _expert(), _expert(), seed=SEED, max_attempts=3
    )
    assert (solved, first_success) == (2, [1, 1])


def test_the_same_seed_gives_the_same_scores() -> None:
    policy = POLICIES["masked_random"]
    runs = [
        score_samples([_board()] * 5, policy, policy, seed=SEED, max_attempts=8)
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def _eval_set(tmp_path: Path) -> Path:
    samples = [_board(), _board()]
    dataset_dir = tmp_path / "datasets" / "tiny"
    dataset_dir.mkdir(parents=True)
    with (dataset_dir / "dataset.pkl").open("wb") as handle:
        pickle.dump({"splits": {"test": samples}}, handle)
    (dataset_dir / MANIFEST_FILENAME).write_text(
        json.dumps({"content_sha256": {"test": split_digest(samples)}}),
        encoding="utf-8",
    )
    export_split(dataset_dir, "test", tmp_path / "eval_set")
    return tmp_path / "eval_set"


def test_a_baseline_run_writes_an_artifact_and_refuses_to_overwrite_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "score",
        "--baseline",
        "masked_random",
        "--eval-set",
        str(_eval_set(tmp_path)),
        "--size",
        "2",
        "--max-attempts",
        "4",
        "--output-dir",
        str(tmp_path / "probes"),
    ]
    artifact = score(build_arg_parser().parse_args(argv))

    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact.name == "cross_size_masked_random_2x2_test.json"
    assert written["puzzles"] == 2
    assert written["judge"] == "strict"
    assert written["checkpoint_sha256"] is None
    assert set(written["best_of_n"]) == {"1", "2", "4"}

    with pytest.raises(SystemExit, match="exists"):
        score(build_arg_parser().parse_args(argv))

    table(
        build_arg_parser().parse_args(
            ["table", "masked_random", "missing", "--size", "2"]
            + ["--output-dir", str(tmp_path / "probes")]
        )
    )
    rows = capsys.readouterr().out.splitlines()
    assert rows[2].startswith("| masked_random | ")
    assert rows[3] == "| missing | (no artifact) |"
