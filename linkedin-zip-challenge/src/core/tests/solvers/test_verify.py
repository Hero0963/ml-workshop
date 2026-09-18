# src/core/tests/solvers/test_verify.py
"""`is_solution` is what stands between a heuristic's best guess and a drawn answer."""

import copy

from src.core.solvers.verify import is_solution
from src.core.tests.conftest import puzzle_01_data, solution_01
from src.core.utils import parse_puzzle_layout


def _board(layout: list[list[str]]) -> dict:
    puzzle = parse_puzzle_layout(layout)
    puzzle["walls"] = set()
    return puzzle


def test_accepts_the_known_solution() -> None:
    assert is_solution(puzzle_01_data, solution_01)


def test_rejects_no_path() -> None:
    assert not is_solution(puzzle_01_data, None)
    assert not is_solution(puzzle_01_data, [])


def test_rejects_a_path_one_cell_short() -> None:
    """The case the heuristics produce most: everything right except coverage."""
    assert not is_solution(puzzle_01_data, solution_01[:-1])


def test_rejects_a_repeated_cell() -> None:
    assert not is_solution(puzzle_01_data, solution_01[:-1] + [solution_01[0]])


def test_rejects_a_jump() -> None:
    jumped = list(solution_01)
    jumped[5], jumped[20] = jumped[20], jumped[5]
    assert not is_solution(puzzle_01_data, jumped)


def test_rejects_a_step_through_a_wall() -> None:
    walled = copy.deepcopy(puzzle_01_data)
    walled["walls"].add(tuple(sorted((solution_01[3], solution_01[4]))))
    assert not is_solution(walled, solution_01)


def test_rejects_numbers_out_of_order() -> None:
    board = _board([["01", "03", "02"]])
    assert is_solution(_board([["01", "02", "03"]]), [(0, 0), (0, 1), (0, 2)])
    assert not is_solution(board, [(0, 0), (0, 1), (0, 2)])


def test_rejects_a_path_that_does_not_start_on_1() -> None:
    assert not is_solution(puzzle_01_data, list(reversed(solution_01)))


def test_rejects_a_step_off_the_grid() -> None:
    """A negative index would silently wrap in the fitness score."""
    board = _board([["01", "02", "03"]])
    assert not is_solution(board, [(0, 0), (0, 1), (-1, 1)])


def test_rejects_a_blocked_cell() -> None:
    board = _board([["01", "  "], ["xx", "02"]])
    assert is_solution(board, [(0, 0), (0, 1), (1, 1)])
    assert not is_solution(board, [(0, 0), (1, 0), (1, 1)])
