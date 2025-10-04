# src/core/solvers/monte_carlo.py


from loguru import logger

from src.core.utils import calculate_fitness_score, generate_random_path


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

        generated_path = generate_random_path(puzzle)
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
