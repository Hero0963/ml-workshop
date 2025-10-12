# src\core\rl\rl_env.py
"""
This module requires the following packages to be installed:
- gymnasium
- numpy
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Optional

from src.core.utils import Puzzle


class PuzzleEnv(gym.Env):
    """
    A custom Gymnasium environment for the pathfinding puzzle.

    The environment's goal is to navigate a grid from a start point, visiting a
    sequence of waypoints in order, and finally reaching an end point.

    **Observation Space**:
    A dictionary with:
    - `agent_location`: The (row, col) of the agent.
    - `next_waypoint_location`: The (row, col) of the next target waypoint.

    **Action Space**:
    A discrete space with 4 actions: 0 (Up), 1 (Down), 2 (Left), 3 (Right).

    **Reward Function (Reward Shaping)**:
    - Large positive reward for reaching the final waypoint.
    - Medium positive reward for reaching a correct intermediate waypoint.
    - Small positive/negative reward based on the change in Manhattan distance
      to the next waypoint (to guide the agent).
    - Negative penalty for each step (to encourage efficiency).
    - Larger negative penalty for invalid moves (hitting a wall or obstacle).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, puzzle: Puzzle, max_steps: int | None = None):
        super().__init__()
        self.puzzle = puzzle
        self.grid_size = puzzle["grid_size"]
        self.walls = puzzle.get("walls", set())
        self.blocked_cells = puzzle.get("blocked_cells", set())

        # Sort waypoints by their number to get the correct sequence
        sorted_waypoints = sorted(puzzle["num_map"].items())
        self.waypoints = [pos for num, pos in sorted_waypoints]
        self.start_pos = self.waypoints[0] if self.waypoints else (0, 0)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0:Up, 1:Down, 2:Left, 3:Right

        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.grid_size[0] - 1, self.grid_size[1] - 1]),
                    dtype=np.int32,
                ),
                "next_waypoint_location": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.grid_size[0] - 1, self.grid_size[1] - 1]),
                    dtype=np.int32,
                ),
            }
        )

        self._agent_location = self.start_pos
        self._next_waypoint_idx = 0
        default_max_steps = self.grid_size[0] * self.grid_size[1] * 2
        self._max_steps = max_steps if max_steps is not None else default_max_steps
        self._current_step = 0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        target_pos = self.waypoints[self._next_waypoint_idx]
        return {
            "agent_location": np.array(self._agent_location, dtype=np.int32),
            "next_waypoint_location": np.array(target_pos, dtype=np.int32),
        }

    def _get_info(self) -> Dict[str, any]:
        target_pos = self.waypoints[self._next_waypoint_idx]
        return {
            "distance_to_next_waypoint": self._calculate_manhattan_distance(
                self._agent_location, target_pos
            )
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
        super().reset(seed=seed)

        self._agent_location = self.start_pos
        self._next_waypoint_idx = 0
        self._current_step = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, any]]:
        self._current_step += 1
        current_pos = self._agent_location
        target_pos = self.waypoints[self._next_waypoint_idx]

        dist_before = self._calculate_manhattan_distance(current_pos, target_pos)

        # --- 1. Determine new position based on action ---
        if action == 0:  # Up
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif action == 1:  # Down
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif action == 2:  # Left
            new_pos = (current_pos[0], current_pos[1] - 1)
        else:  # 3, Right
            new_pos = (current_pos[0], current_pos[1] + 1)

        # --- 2. Check for invalid moves (walls, out of bounds, obstacles) ---
        is_valid_move = True
        if not (
            0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]
        ):
            is_valid_move = False  # Out of bounds
        elif new_pos in self.blocked_cells:
            is_valid_move = False  # Hit an obstacle
        elif tuple(sorted((current_pos, new_pos))) in self.walls:
            is_valid_move = False  # Hit a wall

        if not is_valid_move:
            reward = -10.0  # Penalty for invalid move
            # Agent does not move
            observation = self._get_obs()
            info = self._get_info()
            # An invalid move does not terminate the episode
            return observation, reward, False, False, info

        # --- 3. Valid move: Update agent location and calculate reward ---
        self._agent_location = new_pos
        dist_after = self._calculate_manhattan_distance(new_pos, target_pos)

        # Reward shaping: reward for getting closer to the target
        # NOTE: Weight reduced from 1.0 to 0.1 to test if it resolves looping behavior.
        reward = (dist_before - dist_after) * 0.1
        reward -= 1.0  # Time penalty for each step

        terminated = False

        # --- 4. Check for waypoint events ---
        if self._agent_location == target_pos:
            # Check if this is the final waypoint
            if self._next_waypoint_idx == len(self.waypoints) - 1:
                reward += 1000.0  # Large reward for completing the puzzle
                terminated = True
            else:
                reward += 200.0  # Reward for reaching an intermediate waypoint
                self._next_waypoint_idx += 1

        # --- 5. Check for truncation ---
        truncated = self._current_step >= self._max_steps

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_manhattan_distance(
        self, pos1: Tuple[int, int], pos2: Tuple[int, int]
    ) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self, mode="ansi"):
        # For now, a simple text-based render
        grid_render = [list(row) for row in self.puzzle["puzzle_layout"]]
        agent_r, agent_c = self._agent_location
        grid_render[agent_r][agent_c] = "A "

        rendered_string = "=" * (self.grid_size[1] * 3) + "\n"
        for row in grid_render:
            rendered_string += " ".join(row) + "\n"
        rendered_string += f"Next Waypoint Index: {self._next_waypoint_idx}\n"
        rendered_string += "=" * (self.grid_size[1] * 3) + "\n"
        return rendered_string
