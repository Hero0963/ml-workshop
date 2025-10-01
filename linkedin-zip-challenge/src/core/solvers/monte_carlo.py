# src/core/solvers/monte_carlo.py

import random

from loguru import logger

from src.core.utils import calculate_fitness_score


def _generate_random_path(puzzle: dict) -> list[tuple[int, int]]:
    """Generates a single random path through the puzzle."""
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    num_map = puzzle["num_map"]
    height = len(grid)
    width = len(grid[0])

    if 1 not in num_map:
        return []

    start_pos = num_map[1]
    if start_pos in blocked_cells:
        return []

    path = [start_pos]
    visited = {start_pos}
    current_pos = start_pos

    while True:
        r, c = current_pos
        valid_neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            new_pos = (nr, nc)

            # Standard validity checks
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if new_pos in visited:
                continue
            if new_pos in blocked_cells:
                continue
            wall_pair = tuple(sorted((current_pos, new_pos)))
            if wall_pair in walls:
                continue

            valid_neighbors.append(new_pos)

        if not valid_neighbors:
            break  # Stuck, no valid moves

        # Choose a random next step
        next_pos = random.choice(valid_neighbors)
        path.append(next_pos)
        visited.add(next_pos)
        current_pos = next_pos

    return path


def solve_puzzle_monte_carlo(
    puzzle: dict, attempts: int = 1000
) -> list[tuple[int, int]] | None:
    """
    Solves a puzzle by generating a large number of random paths and
    returning the one with the highest fitness score.

    Args:
        puzzle: The puzzle data dictionary.
        attempts: The number of random paths to generate.

    Returns:
        The best path found, or None if no valid path could be generated.
    """
    best_path: list[tuple[int, int]] | None = None
    best_score = -float("inf")

    # Calculate perfect score once at the beginning. Pass an empty path as it's not used for this calculation.
    _, perfect_score = calculate_fitness_score(puzzle, [])

    logger.info(f"[Monte Carlo] Starting search with {attempts} attempts...")

    for i in range(attempts):
        if (i + 1) % 200 == 0:
            logger.debug(f"[Monte Carlo] Attempt {i + 1}/{attempts}...")

        generated_path = _generate_random_path(puzzle)
        if not generated_path:
            continue

        current_score, _ = calculate_fitness_score(puzzle, generated_path)

        if current_score > best_score:
            best_score = current_score
            best_path = generated_path
            logger.debug(
                f"[Monte Carlo] New best score: {best_score}/{perfect_score} at attempt {i + 1} (path len: {len(best_path)})"
            )

    logger.info(
        f"[Monte Carlo] Search complete. Best score: {best_score}/{perfect_score}, Path length: {len(best_path) if best_path else 0}"
    )
    return best_path
