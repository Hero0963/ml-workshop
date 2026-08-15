# src/core/rl/rl_env_v2.py
"""One-stroke Zip environment (v2) with action masking.

Design decisions (2026-08-15, see `ai-collab/plans/2026-08-15_track-rl-solver.md` §4):

* **Every episode is a one-stroke walk.** Visited cells are masked out, so revisits are
  impossible rather than discouraged. This removes the v1 2-cycle by construction -- an
  oscillation needs a revisit -- so no `visit_count` / `visit_recency` channels are needed.
* **Legality follows `src/core/solvers/dfs.py`**, the project's canonical definition:
  a solution covers every visitable cell and collects the numbers in ascending order
  (`dfs.py:96-105`). Standing on number 1 at reset already collects it (`dfs.py:72-77`);
  v1 got exactly this wrong, which made its terminal state unreachable.
* **Sparse reward.** Success +1, everything else 0, speed expressed through gamma.
  Optional potential-based coverage shaping, off by default in evaluation.
* **Reverse curriculum.** The generator hands back the solution, so an episode can start
  `k` cells from the end with the prefix pre-marked as walked.

The step budget equals the number of unvisited cells, and each step consumes exactly one
of them, so `truncated` is a defensive backstop only: episodes end either in success or
in a dead end (every direction masked).
"""

from typing import Any, NamedTuple, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.rl.action_space import ACTION_DELTAS
from src.core.utils import Puzzle

GRID_PAD = 8
NUM_ACTIONS = 4

CH_VALID = 0
CH_WALL_RIGHT = 1
CH_WALL_DOWN = 2
CH_VISITED = 3
CH_AGENT = 4
CH_WP_NEXT = 5
CH_WP_FUTURE = 6
CH_WP_DONE = 7
NUM_GRID_CHANNELS = 8

# coverage, waypoint progress, last action one-hot(4), height, width
NUM_SCALARS = 8

SUCCESS_REWARD = 1.0
DEFAULT_GAMMA = 0.99
DEFAULT_SHAPING_LAMBDA = 0.2
FIRST_WAYPOINT_NUMBER = 1
MIN_REVERSE_CURRICULUM_K = 2


class PuzzleSample(NamedTuple):
    """A puzzle together with the ground-truth path the generator built it from."""

    puzzle: Puzzle
    solution_path: list[tuple[int, int]]


class PuzzleEnvV2(gym.Env):
    """Gymnasium environment where the agent may never step on a visited cell.

    Observation is a Dict of a padded `GRID_PAD x GRID_PAD` stack and a scalar vector:

    | # | Channel       | Value                                                  |
    |---|---------------|--------------------------------------------------------|
    | 0 | `valid_mask`  | 1 on in-bounds, non-blocked cells                       |
    | 1 | `wall_right`  | 1 when a wall separates the cell from its right neighbour |
    | 2 | `wall_down`   | 1 when a wall separates the cell from the cell below    |
    | 3 | `visited`     | 1 on cells already walked                               |
    | 4 | `agent_pos`   | one-hot                                                 |
    | 5 | `wp_next`     | one-hot of the number that must be collected next       |
    | 6 | `wp_future`   | uncollected numbers marked `number / max_number`        |
    | 7 | `wp_done`     | collected numbers marked 1                              |

    Scalars: coverage, waypoint progress, last action one-hot(4), height and width
    (both normalised by `GRID_PAD`).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        samples: Sequence[PuzzleSample],
        reverse_curriculum_k: int | None = None,
        shaping_lambda: float = DEFAULT_SHAPING_LAMBDA,
        gamma: float = DEFAULT_GAMMA,
    ):
        super().__init__()
        if not samples:
            raise ValueError("PuzzleEnvV2 needs at least one puzzle sample.")

        self.samples = list(samples)
        self.shaping_lambda = shaping_lambda
        self.gamma = gamma
        self.set_reverse_curriculum_k(reverse_curriculum_k)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(NUM_GRID_CHANNELS, GRID_PAD, GRID_PAD),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=0.0, high=1.0, shape=(NUM_SCALARS,), dtype=np.float32
                ),
            }
        )

        self._load_sample(self.samples[0])

    def _load_sample(self, sample: PuzzleSample) -> None:
        """Caches everything that stays constant while this puzzle is being played."""
        puzzle = sample.puzzle
        self.puzzle = puzzle
        self.solution_path = sample.solution_path
        self.height, self.width = puzzle["grid_size"]
        if self.height > GRID_PAD or self.width > GRID_PAD:
            raise ValueError(
                f"Grid {self.height}x{self.width} exceeds the padded observation "
                f"size {GRID_PAD}x{GRID_PAD}."
            )

        self.walls = puzzle.get("walls", set())
        self.blocked_cells = puzzle.get("blocked_cells", set())
        self.num_map: dict[int, tuple[int, int]] = puzzle["num_map"]
        self.max_waypoint_number = max(self.num_map) if self.num_map else 0
        self.waypoint_number_at: dict[tuple[int, int], int] = {
            pos: number for number, pos in self.num_map.items()
        }
        self.visitable_cells = self.height * self.width - len(self.blocked_cells)
        self._static_layers = self._build_static_layers()

    def _build_static_layers(self) -> np.ndarray:
        """valid_mask / wall_right / wall_down never change within a puzzle."""
        layers = np.zeros((3, GRID_PAD, GRID_PAD), dtype=np.float32)
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) in self.blocked_cells:
                    continue
                layers[0, row, col] = 1.0

        for cell_a, cell_b in self.walls:
            (row_a, col_a), (row_b, col_b) = sorted((cell_a, cell_b))
            if row_a == row_b:
                layers[1, row_a, col_a] = 1.0
            else:
                layers[2, row_a, col_a] = 1.0
        return layers

    def set_reverse_curriculum_k(self, k: int | None) -> None:
        """Sets how many cells (including the start) remain when an episode begins.

        `None` starts from the real start cell. `k` must be at least 2, otherwise the
        episode would already be solved at `reset()`.
        """
        if k is not None and k < MIN_REVERSE_CURRICULUM_K:
            raise ValueError(
                f"reverse_curriculum_k must be >= {MIN_REVERSE_CURRICULUM_K} "
                f"(got {k}); k=1 would start on an already-solved board."
            )
        self.reverse_curriculum_k = k

    def _curriculum_start_index(self) -> int:
        if self.reverse_curriculum_k is None:
            return 0
        return max(0, len(self.solution_path) - self.reverse_curriculum_k)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        sample = self.samples[int(self.np_random.integers(len(self.samples)))]
        self._load_sample(sample)

        start_index = self._curriculum_start_index()
        walked = self.solution_path[: start_index + 1]
        self._agent_location = walked[-1]
        self._visited: set[tuple[int, int]] = set(walked)
        # Numbers appear along the solution in ascending order, so the prefix has
        # collected exactly the first `len(prefix numbers)` of them.
        collected = sum(1 for cell in walked if cell in self.waypoint_number_at)
        self._next_waypoint = FIRST_WAYPOINT_NUMBER + collected
        self._last_action: int | None = None
        self._steps = 0
        self._budget = self.visitable_cells - len(self._visited)

        return self._get_obs(), self._get_info()

    def _is_legal_move(self, current: tuple[int, int], target: tuple[int, int]) -> bool:
        row, col = target
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        if target in self.blocked_cells:
            return False
        if target in self._visited:
            return False
        if tuple(sorted((current, target))) in self.walls:
            return False
        number = self.waypoint_number_at.get(target)
        return number is None or number == self._next_waypoint

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the 4 actions, in the shape `MaskablePPO` expects."""
        row, col = self._agent_location
        legal = []
        for action in range(NUM_ACTIONS):
            delta_row, delta_col = ACTION_DELTAS[action]
            legal.append(
                self._is_legal_move(
                    self._agent_location, (row + delta_row, col + delta_col)
                )
            )
        return np.array(legal, dtype=bool)

    def _coverage(self) -> float:
        return len(self._visited) / self.visitable_cells

    def _is_solved(self) -> bool:
        """Matches `dfs.py:96-105`: full coverage plus every number collected in order."""
        return (
            len(self._visited) == self.visitable_cells
            and self._next_waypoint > self.max_waypoint_number
        )

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        masks = self.action_masks()
        if not masks[action]:
            # MaskablePPO never samples a masked action; ending the episode keeps
            # `check_env` usable while making unmasked training obviously fail.
            info = self._get_info()
            info["invalid_action"] = True
            return self._get_obs(), 0.0, True, False, info

        coverage_before = self._coverage()
        delta_row, delta_col = ACTION_DELTAS[action]
        row, col = self._agent_location
        self._agent_location = (row + delta_row, col + delta_col)
        self._visited.add(self._agent_location)
        self._last_action = action
        self._steps += 1

        number = self.waypoint_number_at.get(self._agent_location)
        if number == self._next_waypoint:
            self._next_waypoint += 1

        reward = self.shaping_lambda * (self.gamma * self._coverage() - coverage_before)

        terminated = False
        if self._is_solved():
            reward += SUCCESS_REWARD
            terminated = True
        elif not self.action_masks().any():
            terminated = True  # dead end: every direction is illegal

        truncated = not terminated and self._steps >= self._budget

        info = self._get_info()
        info["dead_end"] = terminated and not self._is_solved()
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        grid = np.zeros((NUM_GRID_CHANNELS, GRID_PAD, GRID_PAD), dtype=np.float32)
        grid[CH_VALID : CH_VALID + 3] = self._static_layers

        for row, col in self._visited:
            grid[CH_VISITED, row, col] = 1.0
        grid[CH_AGENT][self._agent_location] = 1.0

        for number, (row, col) in self.num_map.items():
            if number < self._next_waypoint:
                grid[CH_WP_DONE, row, col] = 1.0
            else:
                grid[CH_WP_FUTURE, row, col] = number / self.max_waypoint_number
        next_pos = self.num_map.get(self._next_waypoint)
        if next_pos is not None:
            grid[CH_WP_NEXT][next_pos] = 1.0

        scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
        scalars[0] = self._coverage()
        scalars[1] = (
            (self._next_waypoint - FIRST_WAYPOINT_NUMBER) / self.max_waypoint_number
            if self.max_waypoint_number
            else 1.0
        )
        if self._last_action is not None:
            scalars[2 + self._last_action] = 1.0
        scalars[6] = self.height / GRID_PAD
        scalars[7] = self.width / GRID_PAD

        return {"grid": grid, "scalars": scalars}

    def _get_info(self) -> dict[str, Any]:
        return {
            "coverage": self._coverage(),
            "next_waypoint": self._next_waypoint,
            "steps": self._steps,
            "solved": self._is_solved(),
            "agent_location": self._agent_location,
        }

    def render(self, mode: str = "ansi") -> str:
        rows = []
        for row in range(self.height):
            cells = []
            for col in range(self.width):
                if (row, col) == self._agent_location:
                    cells.append("A ")
                elif (row, col) in self.blocked_cells:
                    cells.append("xx")
                elif (row, col) in self.waypoint_number_at:
                    cells.append(f"{self.waypoint_number_at[(row, col)]:02d}")
                elif (row, col) in self._visited:
                    cells.append("..")
                else:
                    cells.append("  ")
            rows.append(" ".join(cells))
        header = "=" * (self.width * 3)
        return (
            f"{header}\n"
            + "\n".join(rows)
            + f"\nnext waypoint: {self._next_waypoint}\n{header}\n"
        )
