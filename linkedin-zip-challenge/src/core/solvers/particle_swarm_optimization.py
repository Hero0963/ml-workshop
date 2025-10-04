# src/core/solvers/particle_swarm_optimization.py
import copy
import random
from dataclasses import dataclass, field

from loguru import logger

from src.core.utils import calculate_fitness_score, generate_random_path


@dataclass
class Particle:
    """Represents a particle in the PSO swarm."""

    position: list[tuple[int, int]]  # The path itself
    velocity: list[tuple[int, int]] = field(
        default_factory=list
    )  # List of swap operations
    score: float = -float("inf")
    pbest_position: list[tuple[int, int]] = field(default_factory=list)
    pbest_score: float = -float("inf")

    def __post_init__(self):
        # Initially, the particle's best known position is its starting position
        self.pbest_position = self.position
        self.pbest_score = self.score


def _apply_velocity(
    path: list[tuple[int, int]], velocity: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Applies a velocity (list of swaps) to a path."""
    new_path = list(path)  # Make a mutable copy
    for swap in velocity:
        try:
            idx1, idx2 = new_path.index(swap[0]), new_path.index(swap[1])
            new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        except ValueError:
            # If a point to be swapped isn't in the path, just ignore that swap
            continue
    return new_path


def _subtract_paths(
    path_a: list[tuple[int, int]], path_b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Generates a velocity (list of swaps) to make path_b more like path_a."""
    velocity = []
    # Create a mapping from value to index for faster lookups
    b_map = {val: i for i, val in enumerate(path_b)}

    # Find differences and generate swaps
    for i, val_a in enumerate(path_a):
        if i >= len(path_b):
            break
        val_b = path_b[i]
        if val_a != val_b:
            # val_a is the correct element at this position.
            # Find where val_a is in path_b and swap it with val_b
            if val_a in b_map:
                # Create a swap operation for the values
                velocity.append((val_b, val_a))
                # To keep b_map consistent for the next step, we would need to update it.
                # For simplicity, we generate one swap and return.                return velocity
    return velocity


def solve_puzzle_pso(
    puzzle: dict,
    swarm_size: int = 30,
    num_iterations: int = 50,
    w: float = 0.5,  # Inertia weight
    c1: float = 1.0,  # Cognitive weight
    c2: float = 1.5,  # Social weight
) -> list[tuple[int, int]] | None:
    """
    Solves a puzzle using a discrete Particle Swarm Optimization (PSO) variant.
    """
    _, perfect_score = calculate_fitness_score(puzzle, [])
    logger.info(
        f"[PSO] Starting search with swarm_size={swarm_size}, iterations={num_iterations}"
    )

    # 1. Initialization
    gbest_position = None
    gbest_score = -float("inf")
    swarm = []
    for _ in range(swarm_size):
        path = generate_random_path(puzzle)
        score, _ = calculate_fitness_score(puzzle, path)
        particle = Particle(position=path, score=score)
        swarm.append(particle)

        if particle.pbest_score > gbest_score:
            gbest_score = particle.pbest_score
            gbest_position = particle.pbest_position

    # 2. Main Loop
    for i in range(num_iterations):
        for particle in swarm:
            # 3. Update Velocity
            v_old = particle.velocity
            v_cognitive = _subtract_paths(particle.pbest_position, particle.position)
            v_social = _subtract_paths(gbest_position, particle.position)

            new_velocity = []
            # Inertia component
            if random.random() < w:
                new_velocity.extend(v_old)
            # Cognitive component
            if random.random() < c1:
                new_velocity.extend(v_cognitive)
            # Social component
            if random.random() < c2:
                new_velocity.extend(v_social)

            particle.velocity = new_velocity

            # 4. Update Position
            particle.position = _apply_velocity(particle.position, particle.velocity)

            # 5. Evaluate and Update Bests
            particle.score, _ = calculate_fitness_score(puzzle, particle.position)

            if particle.score > particle.pbest_score:
                particle.pbest_score = particle.score
                particle.pbest_position = particle.position

                if particle.pbest_score > gbest_score:
                    gbest_score = particle.pbest_score
                    gbest_position = copy.deepcopy(particle.pbest_position)

        logger.debug(
            f"[PSO] Iteration {i + 1}/{num_iterations} | Global Best Score: {gbest_score}/{perfect_score}"
        )

    logger.info(
        f"[PSO] Search complete. Best score: {gbest_score}/{perfect_score}, Path length: {len(gbest_position) if gbest_position else 0}"
    )

    return gbest_position
