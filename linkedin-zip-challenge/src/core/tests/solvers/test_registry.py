# src/core/tests/solvers/test_registry.py
"""The solver list must exist once.

It used to exist three times -- API router, vision router, Gradio dropdown -- and the
symptom of that was silent: a solver added to one of them simply did not appear in the
others. These tests fail if a copy comes back.
"""

from src.app.routers import solver as solver_router
from src.app.routers import vision as vision_router
from src.core.solvers import registry
from src.ui import gradio_app


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
        assert entry.kind in {registry.EXACT, registry.LEARNED}
        assert entry.note.strip(), f"{entry.name} has no note"
        assert registry.describe(entry.name) == entry.note


def test_the_exact_solvers_are_the_ones_that_always_answer() -> None:
    """The default a caller falls back to must never be the learned one."""
    assert set(registry.EXACT_SOLVERS) == {"DFS", "A* (heapq)", "CP-SAT"}
    assert vision_router.DEFAULT_SOLVER in registry.EXACT_SOLVERS
