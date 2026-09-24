# src/board_game_rl/agents/mcts_agent.py
"""
Pure Monte Carlo Tree Search (UCT) agent.

Game-agnostic: it needs a `GameRules` forward model but no evaluation function.
Every simulation runs four phases:

1. Selection      - walk down fully expanded nodes, picking the child with the best UCB1
2. Expansion      - add one untried move as a new child
3. Simulation     - play uniformly random moves from that child until the game ends
4. Backpropagation - write the result into every node on the path back to the root

After all simulations, play the root child that was visited the most.
"""

import math
import random
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

from board_game_rl.agents.base import BaseAgent
from board_game_rl.games.base import GameRules

DEFAULT_SIMULATIONS = 1_000
DEFAULT_EXPLORATION = math.sqrt(2)

WIN_SCORE = 1.0
DRAW_SCORE = 0.5
LOSS_SCORE = 0.0

StateT = TypeVar("StateT", bound=Hashable)


@dataclass(eq=False)
class Node(Generic[StateT]):
    """One game state in the search tree.

    `score_sum` is kept from the point of view of `player_just_moved`, the player
    whose move led here. A parent choosing among its children therefore always
    maximises, whichever side it is: the children's scores are already written
    from the chooser's perspective.
    """

    state: StateT
    player_just_moved: int
    parent: "Node[StateT] | None" = None
    action: int | None = None
    untried_actions: list[int] = field(default_factory=list)
    children: list["Node[StateT]"] = field(default_factory=list)
    visits: int = 0
    score_sum: float = 0.0

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.visits if self.visits else 0.0

    def ucb1(self, exploration: float) -> float:
        """Mean score (exploitation) plus a bonus for rarely tried moves (exploration)."""
        parent_visits = self.parent.visits if self.parent else self.visits
        bonus = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return self.mean_score + bonus


class MCTSAgent(BaseAgent, Generic[StateT]):
    """UCT with uniformly random rollouts, no tree reuse and no domain heuristics."""

    def __init__(
        self,
        rules: GameRules[StateT],
        player: int = 1,
        n_simulations: int = DEFAULT_SIMULATIONS,
        exploration: float = DEFAULT_EXPLORATION,
        seed: int | None = None,
        name: str = "MCTS",
    ) -> None:
        super().__init__(name=name)
        if n_simulations < 1:
            raise ValueError("n_simulations must be at least 1.")
        self.rules = rules
        self.player = player
        self.n_simulations = n_simulations
        self.exploration = exploration
        self._rng = random.Random(seed)
        self.last_root: Node[StateT] | None = None

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        state = self.rules.from_observation(observation, to_move=self.player)
        return self.search(state)

    def search(self, state: StateT) -> int:
        """Run `n_simulations` simulations from `state` and return the chosen action."""
        if self.rules.winner(state) is not None:
            raise ValueError("The game is already over.")
        legal_actions = self.rules.legal_actions(state)
        if len(legal_actions) == 1:
            return legal_actions[0]

        root = self._new_node(state, player_just_moved=-self.rules.to_move(state))
        for _ in range(self.n_simulations):
            leaf = self._expand(self._select(root))
            winner = self._rollout(leaf.state)
            self._backpropagate(leaf, winner)

        self.last_root = root
        return max(root.children, key=lambda child: child.visits).action

    def _new_node(
        self,
        state: StateT,
        player_just_moved: int,
        parent: Node[StateT] | None = None,
        action: int | None = None,
    ) -> Node[StateT]:
        untried = (
            self.rules.legal_actions(state) if self.rules.winner(state) is None else []
        )
        self._rng.shuffle(untried)
        return Node(state, player_just_moved, parent, action, untried)

    def _select(self, node: Node[StateT]) -> Node[StateT]:
        while not node.untried_actions and node.children:
            node = max(node.children, key=lambda child: child.ucb1(self.exploration))
        return node

    def _expand(self, node: Node[StateT]) -> Node[StateT]:
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        child = self._new_node(
            self.rules.next_state(node.state, action),
            player_just_moved=self.rules.to_move(node.state),
            parent=node,
            action=action,
        )
        node.children.append(child)
        return child

    def _rollout(self, state: StateT) -> int:
        while (winner := self.rules.winner(state)) is None:
            action = self._rng.choice(self.rules.legal_actions(state))
            state = self.rules.next_state(state, action)
        return winner

    def _backpropagate(self, node: Node[StateT] | None, winner: int) -> None:
        """Walk from `node` up to the root, recording one more visit and the score.

        `winner` is 1 or -1 for the player who won the rollout, or 0 for a draw.
        Each node's `score_sum` is from its own `player_just_moved`'s point of view.
        """
        while node is not None:
            node.visits += 1
            if winner == node.player_just_moved:
                node.score_sum += WIN_SCORE
            elif winner == 0:
                node.score_sum += DRAW_SCORE
            else:
                node.score_sum += LOSS_SCORE
            node = node.parent
