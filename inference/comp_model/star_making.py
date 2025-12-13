"""Minimal Star Making environment for tabular RL/Q-learning.

Episode = 4 actions. After actions 2 and 4, items are created from action pairs.
Two items yield a star. Reward = 1 if final star matches goal, else 0.
"""

import random
from typing import Literal, Tuple

from inference.star_making.assets_utils import StarMakingRules

RuleType = Literal["learning", "transfer"]
State = Tuple[int, int, int, int, int, int, int, int]  # (t, goal_idx, item1, item2, a1, a2, a3, a4)
ITEM_TO_INT = {"A": 0, "B": 1, "C": 2, "D": 3}


class StarMakingEnv:
    """Finite-horizon MDP wrapper around Star Making rules."""

    n_actions = 4
    horizon = 4

    def __init__(self, rule_type: RuleType = "learning", seed: int | None = None):
        self.rules = StarMakingRules()
        self.rule_type = rule_type
        self._rng = random.Random(seed)
        self.goal_space: list[str] = []
        self.goal_star = ""
        self.goal_idx = 0
        self.actions: list[int] = []
        self.reset()

    def reset(self, goal_star: str | None = None) -> State:
        self.actions = []
        tables = self.rules.learning_rules if self.rule_type == "learning" else self.rules.transfer_rules
        self.goal_space = sorted(set(tables["high"].values()))
        self.goal_star = goal_star or self._rng.choice(self.goal_space)
        self.goal_idx = self.goal_space.index(self.goal_star)
        return self.state

    @property
    def state(self) -> State:
        items = self._compute_items()
        item_ints = [ITEM_TO_INT[i] for i in items] + [-1] * (2 - len(items))
        actions = self.actions + [-1] * (4 - len(self.actions))
        return (len(self.actions), self.goal_idx, item_ints[0], item_ints[1], *actions)

    def _compute_items(self) -> list[str]:
        items = []
        if len(self.actions) >= 2:
            if item1 := self.rules.get_item(self.actions[:2], self.rule_type):
                items.append(item1)
            if len(self.actions) >= 4:
                if item2 := self.rules.get_item(self.actions[2:4], self.rule_type):
                    items.append(item2)
        return items

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        if action not in range(4):
            raise ValueError(f"Invalid action {action}")

        self.actions.append(action)
        done = len(self.actions) >= 4
        reward = 0.0

        if done:
            items = self._compute_items()
            if len(items) == 2:
                final_star = self.rules.get_star(items, self.rule_type)
                reward = 1.0 if final_star == self.goal_star else 0.0

        return self.state, reward, done, {}

    def state_index(self, state: State | None = None) -> int:
        """Encode state to integer for tabular Q-learning (base-5 encoding)."""
        s = state or self.state
        t, goal_idx, i1, i2, a1, a2, a3, a4 = s
        idx = t * len(self.goal_space) + goal_idx
        for d in (i1, i2, a1, a2, a3, a4):
            idx = idx * 5 + (d if d != -1 else 4)
        return idx
