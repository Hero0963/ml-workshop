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

import functools
import time
from dataclasses import dataclass
from typing import Callable

from loguru import logger

from src.core.rl.solver_service import solve_puzzle_rl
from src.core.solvers.a_star import solve_puzzle_a_star
from src.core.solvers.ant_colony_optimization import solve_puzzle_ant_colony
from src.core.solvers.cp import solve_puzzle_cp
from src.core.solvers.dfs import solve_puzzle as solve_puzzle_dfs
from src.core.solvers.genetic_algorithm import solve_puzzle_genetic_algorithm
from src.core.solvers.monte_carlo import solve_puzzle_monte_carlo
from src.core.solvers.particle_swarm_optimization import solve_puzzle_pso
from src.core.solvers.simulated_annealing import solve_puzzle_simulated_annealing
from src.core.solvers.tabu_search import solve_puzzle_tabu_search
from src.core.solvers.verify import is_solution
from src.core.utils import Puzzle

SolverFn = Callable[..., list[tuple[int, int]] | None]

EXACT = "exact"
LEARNED = "learned"
#: Not exact and not guaranteed: an answer is checked before it is served, and giving up
#: says nothing about whether the board has a solution.
HEURISTIC = "heuristic"

#: What a heuristic gets per request: reruns until one verifies, for this many seconds.
#: Seconds, because it is the only unit the six share -- their own knobs count random
#: walks, iterations, generations or a temperature schedule -- and it is fixed here rather
#: than taken from the request so that every heuristic is compared on the same budget.
#: Measured and argued in `ai-collab/reports/2026-09-12_heuristic-solvers-on-the-api.md`.
HEURISTIC_TIME_BUDGET_SECONDS = 5.0


def _until_verified(solve: SolverFn) -> SolverFn:
    """Reruns a heuristic until it returns a real solution or the time budget is spent.

    A heuristic returns the best path it saw whether or not that path solves anything, and
    both endpoints draw whatever comes back as the answer. So a run that does not verify is
    retried, and once the budget is gone the answer is `None`, which the endpoints already
    report as "no solution found".
    """
    module = solve.__module__

    @functools.wraps(solve)
    def run(puzzle: Puzzle) -> list[tuple[int, int]] | None:
        started = time.monotonic()
        runs = 0
        path: list[tuple[int, int]] | None = None
        # Every run logs each of its iterations at DEBUG; hundreds of reruns per request
        # would bury the server log and spend the budget on writing it.
        logger.disable(module)
        try:
            while True:
                runs += 1
                path = solve(puzzle)
                if not path or is_solution(puzzle, path):
                    # An empty answer means there is nothing to start from, which no rerun
                    # changes.
                    break
                if time.monotonic() - started >= HEURISTIC_TIME_BUDGET_SECONDS:
                    path = None
                    break
        finally:
            logger.enable(module)

        elapsed = time.monotonic() - started
        if path:
            logger.info(f"[{solve.__name__}] solved on run {runs} in {elapsed:.2f}s")
            return path
        logger.info(f"[{solve.__name__}] gave up after {runs} runs in {elapsed:.2f}s")
        return None

    return run


_HEURISTIC_CAVEAT = (
    "Heuristic: reruns until an answer verifies, for "
    f"{HEURISTIC_TIME_BUDGET_SECONDS:g}s. 'No solution found' means it gave up, not "
    "that the board has none."
)


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
    SolverEntry(
        name="Ant Colony Optimization",
        solve=_until_verified(solve_puzzle_ant_colony),
        kind=HEURISTIC,
        note=f"Ants walk from 1 and reinforce the edges of good walks. {_HEURISTIC_CAVEAT}",
    ),
    SolverEntry(
        name="Genetic Algorithm",
        solve=_until_verified(solve_puzzle_genetic_algorithm),
        kind=HEURISTIC,
        note=f"Keeps the best walks and mutates them, no crossover. {_HEURISTIC_CAVEAT}",
    ),
    SolverEntry(
        name="Particle Swarm Optimization",
        solve=_until_verified(solve_puzzle_pso),
        kind=HEURISTIC,
        note=(
            "Moves walks toward the best one by swapping cells, which breaks them into "
            f"non-adjacent steps -- expect it to give up. {_HEURISTIC_CAVEAT}"
        ),
    ),
    SolverEntry(
        name="Simulated Annealing",
        solve=_until_verified(solve_puzzle_simulated_annealing),
        kind=HEURISTIC,
        note=f"Cuts a walk and regrows it, accepting worse walks early on. {_HEURISTIC_CAVEAT}",
    ),
    SolverEntry(
        name="Tabu Search",
        solve=_until_verified(solve_puzzle_tabu_search),
        kind=HEURISTIC,
        note=f"Local search that refuses to revisit recent walks. {_HEURISTIC_CAVEAT}",
    ),
    SolverEntry(
        name="Monte Carlo",
        solve=_until_verified(solve_puzzle_monte_carlo),
        kind=HEURISTIC,
        note=f"Independent random walks from 1; the baseline. {_HEURISTIC_CAVEAT}",
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
