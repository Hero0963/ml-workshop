# src/core/solvers/verify.py
"""Decides whether a path is a solution, independently of how it was found.

The heuristics optimise `calculate_fitness_score`, so that score cannot also be the judge
of their answers: a path exploiting a gap in the score would be certified by the same gap
(it does not look at blocked cells after the first step, and a negative coordinate indexes
the grid from the far side instead of failing).

The rule is the one the exact solvers stop on -- `dfs.py`'s base case, which
`rl_env_v2._is_solved` mirrors: every open cell exactly once, each step to a neighbour
without crossing a wall, starting on 1, collecting the numbers in order and ending on the
highest one. The end-cell condition was added on 2026-09-19: before that every judge in the
project accepted a walk that passed the last number and kept going, which LinkedIn's help
page leaves open but the generator never produces and rule write-ups exclude.
"""

from src.core.utils import Puzzle


def is_solution(puzzle: Puzzle, path: list[tuple[int, int]] | None) -> bool:
    """True only if `path` solves `puzzle` by the rule the exact solvers use."""
    if not path:
        return False

    height, width = puzzle["grid_size"]
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    num_map = puzzle["num_map"]

    if len(path) != height * width - len(blocked_cells) or len(set(path)) != len(path):
        return False
    for row, col in path:
        if not (0 <= row < height and 0 <= col < width) or (row, col) in blocked_cells:
            return False
    for here, there in zip(path, path[1:]):
        if abs(here[0] - there[0]) + abs(here[1] - there[1]) != 1:
            return False
        if tuple(sorted((here, there))) in walls:
            return False

    if 1 in num_map and path[0] != num_map[1]:
        return False
    if num_map and path[-1] != num_map[max(num_map)]:
        return False
    numbers_in_path_order = [grid[row][col] for row, col in path if grid[row][col] > 0]
    return numbers_in_path_order == sorted(num_map)
