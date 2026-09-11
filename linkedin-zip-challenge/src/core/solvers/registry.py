# src/core/solvers/registry.py
"""The one list of solvers the service offers.

Before this existed the same mapping was written out three times -- `app/routers/solver.py`,
`app/routers/vision.py` and `ui/gradio_app.py` -- so a solver reached the API, the vision
endpoint and the dropdown only if someone remembered all three. The RL solver is the case
that made it worth fixing: it is the first solver that is neither exact nor always
available, so "which solvers exist" and "what is true about each" now travel together.

Adding a solver: write it under `src/core/solvers/`, then add one `SolverEntry` here. The
API, the screenshot endpoint and the Gradio dropdown all pick it up.
"""

from dataclasses import dataclass
from typing import Callable

from src.core.rl.solver_service import solve_puzzle_rl
from src.core.solvers.a_star import solve_puzzle_a_star
from src.core.solvers.cp import solve_puzzle_cp
from src.core.solvers.dfs import solve_puzzle as solve_puzzle_dfs
from src.core.utils import Puzzle

SolverFn = Callable[..., list[tuple[int, int]] | None]

EXACT = "exact"
LEARNED = "learned"


@dataclass(frozen=True)
class SolverEntry:
    """One solver and the two things a caller has to know before choosing it."""

    name: str
    solve: SolverFn
    kind: str
    #: Shown in the UI and the API docs. Say what the trade-off is, not what it does.
    note: str


SOLVER_ENTRIES: tuple[SolverEntry, ...] = (
    SolverEntry(
        name="DFS",
        solve=solve_puzzle_dfs,
        kind=EXACT,
        note="Depth-first search with pruning. Exact; fine on small boards.",
    ),
    SolverEntry(
        name="A* (heapq)",
        solve=solve_puzzle_a_star,
        kind=EXACT,
        note="Best-first search over the same space. Exact.",
    ),
    SolverEntry(
        name="CP-SAT",
        solve=solve_puzzle_cp,
        kind=EXACT,
        note="Constraint solver. Exact and the fastest here -- the default for a reason.",
    ),
    SolverEntry(
        name="RL (behaviour cloning)",
        solve=solve_puzzle_rl,
        kind=LEARNED,
        note=(
            "A trained policy, sampled up to 32 times. Not exact and not guaranteed, and "
            "it needs a checkpoint under models/ (503 without one). One model serves 4x4, "
            "5x5 and 6x6. Trained on boards with 0 or 2-5 walls, so a screenshot with more "
            "than that is out of distribution -- experimental."
        ),
    ),
)

#: Name -> function, which is the shape every call site already used.
SOLVERS: dict[str, SolverFn] = {entry.name: entry.solve for entry in SOLVER_ENTRIES}

#: The solvers that always answer, for defaults and for anything that cannot show an error.
EXACT_SOLVERS: dict[str, SolverFn] = {
    entry.name: entry.solve for entry in SOLVER_ENTRIES if entry.kind == EXACT
}


def describe(name: str) -> str:
    for entry in SOLVER_ENTRIES:
        if entry.name == name:
            return entry.note
    raise KeyError(name)


def solve_with(name: str, puzzle: Puzzle) -> list[tuple[int, int]] | None:
    """Looks a solver up by name and runs it, raising `KeyError` for an unknown name."""
    return SOLVERS[name](puzzle)
