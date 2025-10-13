# src\core\rl\rl_env.py
"""
This module requires the following packages to be installed:
- gymnasium
- numpy
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Optional, List

from src.core.utils import Puzzle


class PuzzleEnv(gym.Env):
    """
    A custom Gymnasium environment for the pathfinding puzzle, modified for CNNs.

    **Observation Space (for CNNs)**:
    A (6, H, W) tensor representing the state of the board:
    - Channel 0: Blocked cells layout (1 for obstacles, 0 otherwise).
    - Channel 1: Horizontal walls (1 on a cell if there is a wall below it).
    - Channel 2: Vertical walls (1 on a cell if there is a wall to its right).
    - Channel 3: Agent's path taken (1 for visited cells, 0 otherwise).
    - Channel 4: Agent's current location (1 at the agent's position).
    - Channel 5: Next waypoint's location (1 at the waypoint's position).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        puzzle: Puzzle,
        max_steps: int | None = None,
        distance_reward_weight: float = 0.01,
    ):
        super().__init__()
        self.puzzle = puzzle
        self.grid_size = puzzle["grid_size"]
        self.height, self.width = self.grid_size
        self.walls = puzzle.get("walls", set())
        self.blocked_cells = puzzle.get("blocked_cells", set())

        sorted_waypoints = sorted(puzzle["num_map"].items())
        self.waypoints = [pos for num, pos in sorted_waypoints]
        self.start_pos = self.waypoints[0] if self.waypoints else (0, 0)

        self.action_space = spaces.Discrete(4)  # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6, self.height, self.width),
            dtype=np.float32,
        )

        # Pre-build static layout layers for efficiency
        self.blocked_cells_layout = self._create_blocked_cells_layout()
        (
            self.horizontal_walls_layout,
            self.vertical_walls_layout,
        ) = self._create_wall_layouts()

        self.total_visitable_cells = self.height * self.width - len(self.blocked_cells)
        self.distance_reward_weight = distance_reward_weight

        self._agent_location = self.start_pos
        self._next_waypoint_idx = 0
        self.path_taken: List[Tuple[int, int]] = []

        default_max_steps = self.grid_size[0] * self.grid_size[1] * 2
        self._max_steps = max_steps if max_steps is not None else default_max_steps
        self._current_step = 0

    def _create_blocked_cells_layout(self) -> np.ndarray:
        """Creates a 2D numpy array for the blocked_cells layout."""
        layout = np.zeros(self.grid_size, dtype=np.float32)
        for r, c in self.blocked_cells:
            layout[r, c] = 1.0
        return layout

    def _create_wall_layouts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates two 2D numpy arrays for horizontal and vertical walls."""
        horizontal_walls = np.zeros(self.grid_size, dtype=np.float32)
        vertical_walls = np.zeros(self.grid_size, dtype=np.float32)

        for wall in self.walls:
            (r1, c1), (r2, c2) = wall
            # Horizontal wall (r1 == r2)
            if r1 == r2:
                # Wall is to the right of the cell with smaller column index
                c = min(c1, c2)
                if c < self.width:
                    vertical_walls[r1, c] = 1.0
            # Vertical wall (c1 == c2)
            else:
                # Wall is below the cell with smaller row index
                r = min(r1, r2)
                if r < self.height:
                    horizontal_walls[r, c1] = 1.0

        return horizontal_walls, vertical_walls

    def _get_obs(self) -> np.ndarray:
        """Constructs the 6-channel observation tensor."""
        # Dynamic layers
        ch_path = np.zeros(self.grid_size, dtype=np.float32)
        for r, c in self.path_taken:
            ch_path[r, c] = 1.0

        ch_agent = np.zeros(self.grid_size, dtype=np.float32)
        ch_agent[self._agent_location] = 1.0

        ch_waypoint = np.zeros(self.grid_size, dtype=np.float32)
        target_pos = self.waypoints[self._next_waypoint_idx]
        ch_waypoint[target_pos] = 1.0

        return np.stack(
            [
                self.blocked_cells_layout,
                self.horizontal_walls_layout,
                self.vertical_walls_layout,
                ch_path,
                ch_agent,
                ch_waypoint,
            ],
            axis=0,
        )

    def _get_info(self) -> Dict[str, any]:
        target_pos = self.waypoints[self._next_waypoint_idx]
        return {
            "distance_to_next_waypoint": self._calculate_manhattan_distance(
                self._agent_location, target_pos
            )
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        super().reset(seed=seed)

        self._agent_location = self.start_pos
        self._next_waypoint_idx = 0
        self._current_step = 0
        self.path_taken = [self.start_pos]

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, any]]:
        self._current_step += 1
        current_pos = self._agent_location
        target_pos = self.waypoints[self._next_waypoint_idx]

        dist_before = self._calculate_manhattan_distance(current_pos, target_pos)

        if action == 0:  # Up
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif action == 1:  # Down
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif action == 2:  # Left
            new_pos = (current_pos[0], current_pos[1] - 1)
        else:  # 3, Right
            new_pos = (current_pos[0], current_pos[1] + 1)

        is_valid_move = True
        if not (0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width):
            is_valid_move = False
        elif new_pos in self.blocked_cells:
            is_valid_move = False
        elif tuple(sorted((current_pos, new_pos))) in self.walls:
            is_valid_move = False

        if not is_valid_move:
            reward = -10.0
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, False, False, info

        # --- 3. Valid move: Update agent location and calculate reward ---
        self._agent_location = new_pos

        # Add penalty for revisiting a cell
        revisit_penalty = 0.0
        if new_pos in self.path_taken:
            revisit_penalty = -2.0  # Penalty for stepping on a previously visited cell

        self.path_taken.append(new_pos)
        dist_after = self._calculate_manhattan_distance(new_pos, target_pos)

        reward = (dist_before - dist_after) * self.distance_reward_weight
        reward -= 1.0  # Time penalty for each step
        reward += revisit_penalty  # Add the new penalty

        terminated = False

        if self._agent_location == target_pos:
            is_final_waypoint = self._next_waypoint_idx == len(self.waypoints) - 1
            all_cells_visited = len(set(self.path_taken)) >= self.total_visitable_cells

            if is_final_waypoint and all_cells_visited:
                reward += 1000.0
                terminated = True
            elif not is_final_waypoint:
                reward += 200.0
                self._next_waypoint_idx += 1

        truncated = self._current_step >= self._max_steps

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_manhattan_distance(
        self, pos1: Tuple[int, int], pos2: Tuple[int, int]
    ) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self, mode="ansi"):
        grid_render = [list(row) for row in self.puzzle["puzzle_layout"]]
        agent_r, agent_c = self._agent_location
        grid_render[agent_r][agent_c] = "A "

        rendered_string = "=" * (self.width * 3) + "\n"
        for row in grid_render:
            rendered_string += " ".join(row) + "\n"
        rendered_string += f"Next Waypoint Index: {self._next_waypoint_idx}\n"
        rendered_string += "=" * (self.width * 3) + "\n"
        return rendered_string
