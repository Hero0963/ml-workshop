# src/core/solvers/ant_colony_optimization.py
import random

from loguru import logger

from src.core.utils import calculate_fitness_score


def _initialize_pheromones(puzzle: dict) -> dict[tuple, float]:
    """Initializes pheromone levels for all valid edges on the grid."""
    grid = puzzle["grid"]
    height = len(grid)
    width = len(grid[0])
    pheromones = {}
    initial_pheromone = 1.0

    for r in range(height):
        for c in range(width):
            current_pos = (r, c)
            for dr, dc in [
                (0, 1),
                (1, 0),
            ]:  # Only check right and down to avoid duplicates
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbor_pos = (nr, nc)
                    edge = tuple(sorted((current_pos, neighbor_pos)))
                    pheromones[edge] = initial_pheromone
    return pheromones


def _build_ant_path(puzzle: dict, pheromones: dict, alpha: float, beta: float) -> list:
    """Builds a single path for one ant based on pheromone trails and heuristics."""
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    num_map = puzzle["num_map"]
    height = len(grid)
    width = len(grid[0])

    start_pos = num_map[1]
    path = [start_pos]
    visited = {start_pos}
    current_pos = start_pos

    while True:
        valid_neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = current_pos[0] + dr, current_pos[1] + dc
            neighbor_pos = (nr, nc)

            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if neighbor_pos in visited or neighbor_pos in blocked_cells:
                continue
            wall_pair = tuple(sorted((current_pos, neighbor_pos)))
            if wall_pair in walls:
                continue
            valid_neighbors.append(neighbor_pos)

        if not valid_neighbors:
            break  # Ant is stuck

        # --- Probabilistic Choice --- #
        attractiveness = []
        heuristic = 1.0  # Simple heuristic

        for neighbor in valid_neighbors:
            edge = tuple(sorted((current_pos, neighbor)))
            pheromone_level = pheromones.get(
                edge, 0.1
            )  # Use a small default if edge not in map
            score = (pheromone_level**alpha) * (heuristic**beta)
            attractiveness.append(score)

        # Choose next path based on weighted probabilities
        total_attractiveness = sum(attractiveness)
        if total_attractiveness == 0:
            # If all attractiveness is zero, choose randomly
            next_pos = random.choice(valid_neighbors)
        else:
            next_pos = random.choices(valid_neighbors, weights=attractiveness, k=1)[0]

        path.append(next_pos)
        visited.add(next_pos)
        current_pos = next_pos

    return path


def solve_puzzle_ant_colony(
    puzzle: dict,
    num_iterations: int = 20,
    num_ants: int = 30,
    evaporation_rate: float = 0.1,
    alpha: float = 1.0,  # Pheromone influence
    beta: float = 1.0,  # Heuristic influence
    pheromone_deposit_amount: float = 100.0,
) -> list[tuple[int, int]] | None:
    """
    Solves a puzzle using the Ant Colony Optimization (ACO) algorithm.
    """
    _, perfect_score = calculate_fitness_score(puzzle, [])
    logger.info(
        f"[ACO] Starting search with {num_iterations} iterations, {num_ants} ants."
    )

    pheromones = _initialize_pheromones(puzzle)
    overall_best_path = None
    overall_best_score = -float("inf")

    for i in range(num_iterations):
        ant_paths = [
            _build_ant_path(puzzle, pheromones, alpha, beta) for _ in range(num_ants)
        ]

        # --- Pheromone Evaporation ---
        for edge in pheromones:
            pheromones[edge] *= 1.0 - evaporation_rate

        # --- Pheromone Deposition ---
        for path in ant_paths:
            if not path:
                continue
            score, _ = calculate_fitness_score(puzzle, path)

            if score > overall_best_score:
                overall_best_score = score
                overall_best_path = path

            # Deposit pheromone based on the quality of the path
            deposit = (score / perfect_score) * pheromone_deposit_amount
            if deposit > 0:
                for j in range(len(path) - 1):
                    edge = tuple(sorted((path[j], path[j + 1])))
                    if edge in pheromones:
                        pheromones[edge] += deposit

        logger.debug(
            f"[ACO] Iteration {i + 1}/{num_iterations} | Best Score: {overall_best_score}/{perfect_score}"
        )

    logger.info(
        f"[ACO] Search complete. Overall best score: {overall_best_score}/{perfect_score}, Path length: {len(overall_best_path) if overall_best_path else 0}"
    )

    return overall_best_path
