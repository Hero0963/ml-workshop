# src/core/solvers/a_star.py

import heapq
import itertools

from loguru import logger
from sortedcontainers import SortedList


# Heuristic function (Manhattan distance to the next waypoint)
def _heuristic(
    current_pos: tuple[int, int],
    next_waypoint_pos: tuple[int, int] | None,
) -> int:
    if next_waypoint_pos is None:
        return 0
    return abs(current_pos[0] - next_waypoint_pos[0]) + abs(
        current_pos[1] - next_waypoint_pos[1]
    )


def solve_puzzle_a_star(puzzle: dict) -> list[tuple[int, int]] | None:
    """
    Solves a Zip puzzle using the A* search algorithm with heapq.
    """
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    height = len(grid)
    width = len(grid[0])
    visitable_cells = (height * width) - len(blocked_cells)
    num_map = puzzle["num_map"]

    if 1 not in num_map:
        return None  # No starting point

    start_pos = num_map[1]
    if start_pos in blocked_cells:
        return None

    # The priority queue will store tuples of:
    # (f_cost, unique_counter, g_cost, path, visited_set, next_waypoint_num)
    # f_cost = g_cost + h_cost
    # g_cost is the length of the path
    # unique_counter is used to make tuples unique for the priority queue
    counter = itertools.count()

    g_cost = 1
    next_waypoint_num = 2
    next_waypoint_pos = num_map.get(next_waypoint_num)
    h_cost = _heuristic(start_pos, next_waypoint_pos)
    f_cost = g_cost + h_cost

    path = [start_pos]
    visited = {start_pos}

    pq = [(f_cost, next(counter), g_cost, path, visited, next_waypoint_num)]

    closed_set = set()

    while pq:
        (
            f,
            _,
            g,
            current_path,
            current_visited,
            current_waypoint_num,
        ) = heapq.heappop(pq)

        last_pos = current_path[-1]

        state_key = (last_pos, frozenset(current_visited))
        if state_key in closed_set:
            continue
        closed_set.add(state_key)

        logger.trace(
            f"A* exploring path of length {len(current_path)}, f={f}, g={g}, h={f-g}"
        )

        if len(current_path) == visitable_cells:
            if not num_map or current_waypoint_num > max(num_map.keys()):
                logger.debug("A* SUCCESS: Found a full path.")
                return current_path
            else:
                continue

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = last_pos[0] + dr, last_pos[1] + dc
            new_pos = (nr, nc)

            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if new_pos in current_visited:
                continue
            if new_pos in blocked_cells:
                continue
            wall_pair = tuple(sorted((last_pos, new_pos)))
            if wall_pair in walls:
                continue

            num_at_new_pos = grid[nr][nc]

            if num_at_new_pos > 0 and num_at_new_pos != current_waypoint_num:
                continue

            new_path = current_path + [new_pos]
            new_visited = current_visited | {new_pos}
            new_g_cost = g + 1

            new_waypoint_num = (
                current_waypoint_num + 1
                if num_at_new_pos == current_waypoint_num
                else current_waypoint_num
            )

            next_waypoint_pos = num_map.get(new_waypoint_num)
            new_h_cost = _heuristic(new_pos, next_waypoint_pos)
            new_f_cost = new_g_cost + new_h_cost

            heapq.heappush(
                pq,
                (
                    new_f_cost,
                    next(counter),
                    new_g_cost,
                    new_path,
                    new_visited,
                    new_waypoint_num,
                ),
            )

    return None


def solve_puzzle_a_star_sortedlist(puzzle: dict) -> list[tuple[int, int]] | None:
    """
    Solves a Zip puzzle using the A* search algorithm with SortedList.
    """
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    height = len(grid)
    width = len(grid[0])
    visitable_cells = (height * width) - len(blocked_cells)
    num_map = puzzle["num_map"]
    if 1 not in num_map:
        return None

    start_pos = num_map[1]
    if start_pos in blocked_cells:
        return None

    counter = itertools.count()
    g_cost = 1
    next_waypoint_num = 2
    next_waypoint_pos = num_map.get(next_waypoint_num)
    h_cost = _heuristic(start_pos, next_waypoint_pos)
    f_cost = g_cost + h_cost

    path = [start_pos]
    visited = {start_pos}

    # Use SortedList as the priority queue
    open_set = SortedList(
        [(f_cost, next(counter), g_cost, path, visited, next_waypoint_num)]
    )
    closed_set = set()

    while open_set:
        (
            f,
            _,
            g,
            current_path,
            current_visited,
            current_waypoint_num,
        ) = open_set.pop(0)  # Pop the element with the smallest f_cost

        last_pos = current_path[-1]

        state_key = (last_pos, frozenset(current_visited))
        if state_key in closed_set:
            continue
        closed_set.add(state_key)

        logger.trace(
            f"A* (SortedList) exploring path of length {len(current_path)}, f={f}, g={g}, h={f-g}"
        )

        if len(current_path) == visitable_cells:
            if not num_map or current_waypoint_num > max(num_map.keys()):
                logger.debug("A* (SortedList) SUCCESS: Found a full path.")
                return current_path
            else:
                continue

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = last_pos[0] + dr, last_pos[1] + dc
            new_pos = (nr, nc)

            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if new_pos in current_visited:
                continue
            if new_pos in blocked_cells:
                continue
            wall_pair = tuple(sorted((last_pos, new_pos)))
            if wall_pair in walls:
                continue

            num_at_new_pos = grid[nr][nc]

            if num_at_new_pos > 0 and num_at_new_pos != current_waypoint_num:
                continue

            new_path = current_path + [new_pos]
            new_visited = current_visited | {new_pos}
            new_g_cost = g + 1

            new_waypoint_num = (
                current_waypoint_num + 1
                if num_at_new_pos == current_waypoint_num
                else current_waypoint_num
            )

            next_waypoint_pos = num_map.get(new_waypoint_num)
            new_h_cost = _heuristic(new_pos, next_waypoint_pos)
            new_f_cost = new_g_cost + new_h_cost

            open_set.add(
                (
                    new_f_cost,
                    next(counter),
                    new_g_cost,
                    new_path,
                    new_visited,
                    new_waypoint_num,
                )
            )

    return None
