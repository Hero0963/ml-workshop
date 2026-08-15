# src/core/rl/action_space.py
"""Shared action encoding for the Zip RL environments.

The legacy `PuzzleEnv` hard-codes the mapping 0:Up, 1:Down, 2:Left, 3:Right.
Centralising it here keeps replay helpers, diagnostics and the upcoming env v2
from re-deriving the same convention.
"""

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_DELTAS: dict[int, tuple[int, int]] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

DELTA_TO_ACTION: dict[tuple[int, int], int] = {
    delta: action for action, delta in ACTION_DELTAS.items()
}


def path_to_actions(path: list[tuple[int, int]]) -> list[int]:
    """Converts a cell-by-cell path into the action indices the env expects.

    Raises:
        ValueError: if two consecutive cells are not orthogonal neighbours.
    """
    actions: list[int] = []
    for current, following in zip(path, path[1:]):
        delta = (following[0] - current[0], following[1] - current[1])
        if delta not in DELTA_TO_ACTION:
            raise ValueError(
                f"Path segment {current} -> {following} is not a single orthogonal step."
            )
        actions.append(DELTA_TO_ACTION[delta])
    return actions
