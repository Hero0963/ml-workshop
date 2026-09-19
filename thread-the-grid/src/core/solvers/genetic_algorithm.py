# src/core/solvers/genetic_algorithm.py
import random

from loguru import logger

from src.core.utils import (
    calculate_fitness_score,
    generate_neighbor_path,
    generate_random_path,
)


def solve_puzzle_genetic_algorithm(
    puzzle: dict,
    population_size: int = 100,
    num_generations: int = 50,
    num_elites: int = 10,
) -> list[tuple[int, int]] | None:
    """
    Solves a puzzle using a Genetic Algorithm variant.

    This implementation uses elitism and mutation, but forgoes a complex
    crossover operation to ensure all paths remain valid on the grid.

    Args:
        puzzle: The puzzle data dictionary.
        population_size: The number of individuals (paths) in each generation.
        num_generations: The number of generations to run the simulation for.
        num_elites: The number of top individuals to carry over to the next generation.

    Returns:
        The best path found across all generations.
    """
    _, perfect_score = calculate_fitness_score(puzzle, [])
    logger.info(
        f"[GA] Starting search with pop_size={population_size}, generations={num_generations}, elites={num_elites}"
    )

    # 1. Initialization
    population = [generate_random_path(puzzle) for _ in range(population_size)]

    overall_best_path = None
    overall_best_score = -float("inf")

    # 2. Generation Loop
    for gen in range(num_generations):
        # 3. Evaluation
        scored_population = [
            (calculate_fitness_score(puzzle, path)[0], path) for path in population
        ]
        scored_population.sort(key=lambda x: x[0], reverse=True)

        current_best_score, current_best_path = scored_population[0]
        if current_best_score > overall_best_score:
            overall_best_score = current_best_score
            overall_best_path = current_best_path

        logger.debug(
            f"[GA] Generation {gen + 1}/{num_generations} | Best Score: {current_best_score}/{perfect_score}"
        )

        # Prepare for the next generation
        next_population = []

        # 4. Selection (Elitism)
        elites = [path for _, path in scored_population[:num_elites]]
        next_population.extend(elites)

        # 5. Reproduction (via Mutation)
        num_offspring = population_size - num_elites
        for _ in range(num_offspring):
            # Select a random elite parent to mutate
            parent = random.choice(elites)
            offspring = generate_neighbor_path(parent, puzzle)
            next_population.append(offspring)

        # 6. Replacement
        population = next_population

    logger.info(
        f"[GA] Search complete. Overall best score: {overall_best_score}/{perfect_score}, Path length: {len(overall_best_path) if overall_best_path else 0}"
    )

    return overall_best_path
