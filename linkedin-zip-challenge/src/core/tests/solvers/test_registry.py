# src/core/tests/solvers/test_registry.py
"""The solver list must exist once.

It used to exist three times -- API router, vision router, Gradio dropdown -- and the
symptom of that was silent: a solver added to one of them simply did not appear in the
others. These tests fail if a copy comes back.
"""

import time

from src.app.routers import solver as solver_router
from src.app.routers import vision as vision_router
from src.core.solvers import registry
from src.core.tests.conftest import puzzle_01_data, solution_01
from src.ui import gradio_app

HEURISTIC_NAMES = {
    "Ant Colony Optimization",
    "Genetic Algorithm",
    "Particle Swarm Optimization",
    "Simulated Annealing",
    "Tabu Search",
    "Monte Carlo",
}
SHORT_BUDGET_SECONDS = 0.05
#: Slack over the budget for the one run that may start just before it runs out.
BUDGET_SLACK_SECONDS = 1.0


def test_every_entry_point_shares_one_registry() -> None:
    assert solver_router.SOLVERS is registry.SOLVERS
    assert vision_router.SOLVERS is registry.SOLVERS
    assert gradio_app.SOLVERS is registry.SOLVERS


def test_the_registry_has_no_duplicate_names() -> None:
    names = [entry.name for entry in registry.SOLVER_ENTRIES]
    assert len(names) == len(set(names))
    assert set(names) == set(registry.SOLVERS)


def test_every_solver_declares_a_kind_and_a_note() -> None:
    for entry in registry.SOLVER_ENTRIES:
        assert entry.kind in {registry.EXACT, registry.LEARNED, registry.HEURISTIC}
        assert entry.note.strip(), f"{entry.name} has no note"
        assert registry.describe(entry.name) == entry.note


def test_the_exact_solvers_are_the_ones_that_always_answer() -> None:
    """The default a caller falls back to must never be the learned one."""
    assert set(registry.EXACT_SOLVERS) == {"DFS", "A* (heapq)", "CP-SAT"}
    assert vision_router.DEFAULT_SOLVER in registry.EXACT_SOLVERS


def test_all_six_heuristics_are_served_as_heuristics() -> None:
    served = {
        entry.name
        for entry in registry.SOLVER_ENTRIES
        if entry.kind == registry.HEURISTIC
    }
    assert served == HEURISTIC_NAMES


def test_no_heuristic_is_served_without_verification() -> None:
    """A raw heuristic returns its best guess, and the endpoints would draw it."""
    for entry in registry.SOLVER_ENTRIES:
        if entry.kind == registry.HEURISTIC:
            assert hasattr(entry.solve, "__wrapped__"), entry.name


class TestUntilVerified:
    def test_a_best_guess_that_is_not_a_solution_becomes_none(self, monkeypatch):
        monkeypatch.setattr(
            registry, "HEURISTIC_TIME_BUDGET_SECONDS", SHORT_BUDGET_SECONDS
        )
        solve = registry._until_verified(lambda puzzle: solution_01[:-1])

        assert solve(puzzle_01_data) is None

    def test_reruns_until_a_run_verifies(self):
        answers = iter([solution_01[:-1], solution_01[:-2], solution_01])
        calls = []

        def flaky(puzzle):
            calls.append(puzzle)
            return next(answers)

        assert registry._until_verified(flaky)(puzzle_01_data) == solution_01
        assert len(calls) == 3

    def test_stops_at_once_when_there_is_nothing_to_start_from(self):
        calls = []

        def empty(puzzle):
            calls.append(puzzle)
            return []

        assert registry._until_verified(empty)(puzzle_01_data) is None
        assert len(calls) == 1

    def test_gives_up_when_the_budget_is_spent(self, monkeypatch):
        monkeypatch.setattr(
            registry, "HEURISTIC_TIME_BUDGET_SECONDS", SHORT_BUDGET_SECONDS
        )
        solve = registry._until_verified(lambda puzzle: solution_01[:-1])

        started = time.monotonic()
        solve(puzzle_01_data)

        assert time.monotonic() - started < SHORT_BUDGET_SECONDS + BUDGET_SLACK_SECONDS
