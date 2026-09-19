# src/core/solvers/tabu_search.py
import collections

from loguru import logger

from src.core.utils import (
    calculate_fitness_score,
    generate_neighbor_path,
    generate_random_path,
)


def solve_puzzle_tabu_search(
    puzzle: dict,
    num_iterations: int = 100,
    tabu_list_size: int = 15,
    neighborhood_size: int = 20,
    aspiration_threshold: float = 0.95,
) -> list[tuple[int, int]] | None:
    """
    Solves a puzzle using the Tabu Search metaheuristic.

    Args:
        puzzle: The puzzle data dictionary.
        num_iterations: The total number of iterations to run.
        tabu_list_size: The size of the tabu list (short-term memory).
        neighborhood_size: The number of neighbors to generate in each iteration.
        aspiration_threshold: A factor (0.0 to 1.0) to determine the aspiration
            criterion. A tabu move is pardoned if its score is >= this value
            times the best score found so far.

    Returns:
        The best path found during the search.
    """
    _, perfect_score = calculate_fitness_score(puzzle, [])
    logger.info(
        f"[TS] Starting search with iterations={num_iterations}, tabu_size={tabu_list_size}"
    )

    # 1. Start with an initial solution
    current_solution = generate_random_path(puzzle)
    if not current_solution:
        logger.warning("[TS] Could not generate an initial random path.")
        return None

    best_solution = current_solution
    best_score, _ = calculate_fitness_score(puzzle, best_solution)

    # 2. Initialize Tabu List
    # A deque with maxlen is an efficient fixed-size queue.
    # We store hashes of the paths to save memory.
    tabu_list = collections.deque(maxlen=tabu_list_size)
    tabu_list.append(hash(tuple(current_solution)))

    # 3. Main Search Loop
    for i in range(num_iterations):
        # 4. Generate Neighborhood
        neighborhood = [
            generate_neighbor_path(current_solution, puzzle)
            for _ in range(neighborhood_size)
        ]

        # 5. Evaluate Neighbors
        scored_neighbors = [
            (calculate_fitness_score(puzzle, path)[0], path) for path in neighborhood
        ]
        scored_neighbors.sort(key=lambda x: x[0], reverse=True)

        # 6. Find Best Non-Tabu Neighbor
        best_candidate = None
        best_candidate_score = -float("inf")

        for score, path in scored_neighbors:
            path_hash = hash(tuple(path))

            # Aspiration Criterion: Pardon a tabu move if it's "good enough".
            is_tabu = path_hash in tabu_list
            is_pardoned = score >= aspiration_threshold * best_score

            if not is_tabu or is_pardoned:
                best_candidate = path
                best_candidate_score = score
                break  # Move to the best valid neighbor

        # If all neighbors are tabu and not pardoned, end the search early.
        if best_candidate is None:
            logger.warning("[TS] Stuck. All neighbors are tabu. Stopping early.")
            break

        # 7. Move to the new solution
        current_solution = best_candidate
        current_score = best_candidate_score

        # 8. Update Tabu List and Best Solution
        tabu_list.append(hash(tuple(current_solution)))

        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score

        logger.debug(
            f"[TS] Iteration {i + 1}/{num_iterations} | Current Score: {current_score}/{perfect_score} | Best Score: {best_score}/{perfect_score}"
        )

    logger.info(
        f"[TS] Search complete. Best score: {best_score}/{perfect_score}, Path length: {len(best_solution) if best_solution else 0}"
    )

    return best_solution
