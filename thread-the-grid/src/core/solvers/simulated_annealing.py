# src/core/solvers/simulated_annealing.py
import math
import random

from loguru import logger

from src.core.utils import (
    calculate_fitness_score,
    generate_neighbor_path,
    generate_random_path,
)


def solve_puzzle_simulated_annealing(
    puzzle: dict,
    initial_temp: float = 10000.0,
    final_temp: float = 0.1,
    cooling_rate: float = 0.995,
) -> list[tuple[int, int]] | None:
    """
    Solves a puzzle using the Simulated Annealing metaheuristic.

    Args:
        puzzle: The puzzle data dictionary.
        initial_temp: The starting temperature for the annealing process.
        final_temp: The temperature at which to stop the process.
        cooling_rate: The factor by which the temperature is reduced in each iteration.

    Returns:
        The best path found, or None if no valid path could be generated.
    """
    _, perfect_score = calculate_fitness_score(puzzle, [])
    logger.info(
        f"[SA] Starting search with T_initial={initial_temp}, T_final={final_temp}, alpha={cooling_rate}"
    )

    current_path = generate_random_path(puzzle)
    if not current_path:
        logger.warning("[SA] Could not generate an initial random path.")
        return None

    current_score, _ = calculate_fitness_score(puzzle, current_path)
    best_path = current_path
    best_score = current_score
    temp = initial_temp
    iteration = 0

    while temp > final_temp:
        iteration += 1
        # Generate a neighbor path (passing the puzzle context is now required)
        neighbor_path = generate_neighbor_path(current_path, puzzle)
        neighbor_score, _ = calculate_fitness_score(puzzle, neighbor_path)

        # We use fitness score as energy, but since higher score is better,
        # we check for neighbor_score > current_score. The acceptance
        # probability for a worse solution is exp((neighbor - current) / T).
        score_diff = neighbor_score - current_score

        if score_diff > 0 or math.exp(score_diff / temp) > random.random():
            current_path = neighbor_path
            current_score = neighbor_score

            if current_score > best_score:
                best_score = current_score
                best_path = current_path
                logger.debug(
                    f"[SA] Iter {iteration}: New best score: {best_score}/{perfect_score} at T={temp:.2f}"
                )

        temp *= cooling_rate

    logger.info(
        f"[SA] Search complete after {iteration} iterations. Best score: {best_score}/{perfect_score}, Path length: {len(best_path)}"
    )
    return best_path
