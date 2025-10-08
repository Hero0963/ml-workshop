# src/core/puzzle_generator.py
import random
import time
from loguru import logger

from src.core.utils import Puzzle, parse_puzzle_layout

# --- Constants ---
DEFAULT_MIN_WALLS = 2
DEFAULT_MAX_WALLS = 5
MAX_RETRIES_PER_COUNT = 5


def generate_puzzle(
    m: int = 7,
    n: int = 7,
    has_walls: bool = False,
    num_blocked_cells: int = 0,
    num_waypoints: int | None = None,
    timeout_per_attempt: float = 20.0,  # Add timeout for each attempt
) -> tuple[Puzzle, list[tuple[int, int]]] | None:
    """
    Generates a new puzzle with a guaranteed solution.
    """
    if num_blocked_cells >= m * n:
        logger.error(
            "Number of blocked cells cannot be equal to or greater than total cells."
        )
        return None

    solution_path = None
    blocked_cells = set()

    for current_blocked_count in range(num_blocked_cells, -1, -1):
        logger.info(
            f"Attempting to generate puzzle with {current_blocked_count} blocked cells."
        )
        for attempt in range(MAX_RETRIES_PER_COUNT):
            attempt_start_time = time.time()
            all_cells = [(r, c) for r in range(m) for c in range(n)]
            current_blocked_cells = set(random.sample(all_cells, current_blocked_count))

            path = _generate_hamiltonian_path(
                m,
                n,
                current_blocked_cells,
                start_time=attempt_start_time,
                timeout=timeout_per_attempt,
            )

            if path:
                logger.success(
                    f"Successfully generated path with {current_blocked_count} blocked cells on attempt {attempt + 1}."
                )
                solution_path = path
                blocked_cells = current_blocked_cells
                break
            else:
                # This log now covers both normal failures and timeouts
                logger.warning(
                    f"Failed to generate path with {current_blocked_count} blocked cells (Attempt {attempt + 1}/{MAX_RETRIES_PER_COUNT})."
                )

        if solution_path:
            break

    if not solution_path:
        logger.error(
            "Failed to generate a valid puzzle even after all retries and decrements."
        )
        return None

    # 3. Define Waypoints
    if num_waypoints is None:
        min_wp = len(solution_path) // 4
        max_wp = len(solution_path) // 3
        if max_wp < 2:
            max_wp = 2
        if min_wp < 2:
            min_wp = 2
        num_waypoints = random.randint(min_wp, max_wp)

    if len(solution_path) < num_waypoints:
        num_waypoints = len(solution_path)

    if num_waypoints < 2 and len(solution_path) >= 2:
        num_waypoints = 2
    elif len(solution_path) < 2:
        num_waypoints = len(solution_path)

    if num_waypoints > 2:
        waypoint_indices = sorted(
            random.sample(range(1, len(solution_path) - 1), num_waypoints - 2)
        )
        final_waypoint_indices = [0] + waypoint_indices + [len(solution_path) - 1]
    elif num_waypoints == 2:
        final_waypoint_indices = [0, len(solution_path) - 1]
    else:
        final_waypoint_indices = list(range(len(solution_path)))

    num_map = {
        i + 1: solution_path[idx] for i, idx in enumerate(final_waypoint_indices)
    }

    # 4. Add Walls
    walls = set()
    if has_walls:
        possible_walls = set()
        for r in range(m):
            for c in range(n - 1):
                possible_walls.add(tuple(sorted(((r, c), (r, c + 1)))))
        for r in range(m - 1):
            for c in range(n):
                possible_walls.add(tuple(sorted(((r, c), (r + 1, c)))))

        solution_edges = {
            tuple(sorted((solution_path[i], solution_path[i + 1])))
            for i in range(len(solution_path) - 1)
        }
        safe_walls = possible_walls - solution_edges

        num_walls = random.randint(DEFAULT_MIN_WALLS, DEFAULT_MAX_WALLS)
        if len(safe_walls) >= num_walls:
            walls = set(random.sample(list(safe_walls), num_walls))

    # 5. Format Puzzle using the canonical parser
    puzzle_layout = [["  " for _ in range(n)] for _ in range(m)]
    for r, c in blocked_cells:
        puzzle_layout[r][c] = "xx"
    for num, pos in num_map.items():
        puzzle_layout[pos[0]][pos[1]] = f"{num:02d}"

    puzzle = parse_puzzle_layout(puzzle_layout)
    puzzle["walls"] = walls

    return puzzle, solution_path


def _generate_hamiltonian_path(
    m: int,
    n: int,
    blocked_cells: set[tuple[int, int]],
    start_time: float,
    timeout: float,
) -> list[tuple[int, int]] | None:
    """
    Generates a Hamiltonian path on a grid with blocked cells using randomized backtracking.
    Includes a timeout mechanism to prevent excessively long runs.
    """
    visitable_cells = {
        (r, c) for r in range(m) for c in range(n) if (r, c) not in blocked_cells
    }
    if not visitable_cells:
        return None

    path = []
    start_cell = random.choice(list(visitable_cells))
    visited = set()

    def solve(r: int, c: int) -> bool:
        """Recursively find a path."""
        if time.time() - start_time > timeout:
            return False  # Timeout exceeded

        path.append((r, c))
        visited.add((r, c))

        if len(visited) == len(visitable_cells):
            return True

        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(neighbors)

        for nr, nc in neighbors:
            if (nr, nc) in visitable_cells and (nr, nc) not in visited:
                if solve(nr, nc):
                    return True

        path.pop()
        visited.remove((r, c))
        return False

    if solve(start_cell[0], start_cell[1]):
        return path
    return None
