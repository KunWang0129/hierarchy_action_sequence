"""Very small flat-sequence Q-learning agent.

"Flat" means each 4-action sequence is treated as one bandit arm:
Q(seq) is updated from the final reward, and the next sequence is sampled from
a softmax policy P(seq) ∝ exp(beta * Q(seq)).
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

ActionSeq = Tuple[int, ...]


def _softmax(values: Sequence[float], beta: float) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(beta * (v - m)) for v in values]
    z = sum(exps)
    return [e / z for e in exps]


@dataclass
class FlatQAgent:
    """Tabular flat-sequence learner with softmax action selection."""

    alpha: float = 0.3  # learning rate
    beta: float = 50  # inverse temperature
    horizon: int = 4
    n_actions: int = 4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.sequences: List[ActionSeq] = list(
            itertools.product(range(self.n_actions), repeat=self.horizon)
        )
        self.q: Dict[ActionSeq, float] = {seq: 0.0 for seq in self.sequences}

    def select_sequence(self) -> ActionSeq:
        probs = _softmax([self.q[s] for s in self.sequences], self.beta)
        r = self._rng.random()
        c = 0.0
        for seq, p in zip(self.sequences, probs):
            c += p
            if r <= c:
                return seq
        return self.sequences[-1]

    def update(self, seq: ActionSeq, reward: float) -> None:
        self.q[seq] += self.alpha * (reward - self.q[seq])

    def run_trial(self, env: Any, goal_star: Optional[str] = None) -> Dict[str, Any]:
        """Sample a sequence, execute it, update Q, and return a small record."""
        if goal_star is None:
            env.reset()
        else:
            env.reset(goal_star=goal_star)

        seq = self.select_sequence()
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        for a in seq:
            _, r, done, info = env.step(a)
            reward += float(r)
            if done:
                break

        self.update(seq, reward)
        return {"sequence": seq, "reward": reward, "done": done, "info": info}


@dataclass
class HierarchicalQAgent:
    """Item-based hierarchical learner with two-level Q-tables."""

    alpha_pair: float = 0.8  # Low-level learning rate
    alpha_item: float = 0.2  # High-level learning rate
    beta_pair: float = 3 # Inverse temperature
    lambda_: float = 5.0  # Item-level value weight
    n_actions: int = 4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        # All 16 action pairs
        self.pairs: List[ActionSeq] = list(
            itertools.product(range(self.n_actions), repeat=2)
        )
        # Low-level Q-table: action pair -> item existence value
        self.q_pair: Dict[ActionSeq, float] = {p: 0.0 for p in self.pairs}
        # High-level Q-table: item pair -> final reward value
        items = ["A", "B", "C", "D", None]
        self.q_item: Dict[Tuple[Optional[str], Optional[str]], float] = {
            (i1, i2): 0.0 for i1 in items for i2 in items
        }

    def select_sequence(self, env: Any) -> ActionSeq:
        """Two-stage selection: first pair independent, second pair conditioned on first item."""
        # Stage 1: Select first pair from Q_pair
        values = [self.q_pair[p] for p in self.pairs]
        probs = _softmax(values, self.beta_pair)
        r = self._rng.random()
        c = 0.0
        p1 = self.pairs[0]
        for p, prob in zip(self.pairs, probs):
            c += prob
            if r <= c:
                p1 = p
                break

        # Get item from first pair
        i1 = env.rules.get_item(list(p1), env.rule_type)

        # Stage 2: Select second pair conditioned on i1
        combined_values = []
        for p in self.pairs:
            i2 = env.rules.get_item(list(p), env.rule_type)
            q_tilde = self.q_pair[p] + self.lambda_ * self.q_item[(i1, i2)]
            combined_values.append(q_tilde)

        probs = _softmax(combined_values, self.beta_pair)
        r = self._rng.random()
        c = 0.0
        p2 = self.pairs[0]
        for p, prob in zip(self.pairs, probs):
            c += prob
            if r <= c:
                p2 = p
                break

        return (p1[0], p1[1], p2[0], p2[1])

    def update(self, env: Any, seq: ActionSeq, reward: float) -> None:
        """Two-level learning: update Q_pair and Q_item."""
        # Extract pairs
        p1 = (seq[0], seq[1])
        p2 = (seq[2], seq[3])

        # Low-level updates: pair -> item existence
        i1 = env.rules.get_item(list(p1), env.rule_type)
        i2 = env.rules.get_item(list(p2), env.rule_type)

        r_item_1 = 1.0 if i1 is not None else 0.0
        r_item_2 = 1.0 if i2 is not None else 0.0

        self.q_pair[p1] += self.alpha_pair * (r_item_1 - self.q_pair[p1])
        self.q_pair[p2] += self.alpha_pair * (r_item_2 - self.q_pair[p2])

        # High-level update: item pair -> final reward
        self.q_item[(i1, i2)] += self.alpha_item * (reward - self.q_item[(i1, i2)])

    def run_trial(self, env: Any, goal_star: Optional[str] = None) -> Dict[str, Any]:
        """Sample a sequence, execute it, update Q-tables, and return a record."""
        if goal_star is None:
            env.reset()
        else:
            env.reset(goal_star=goal_star)

        seq = self.select_sequence(env)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        for a in seq:
            _, r, done, info = env.step(a)
            reward += float(r)
            if done:
                break

        self.update(env, seq, reward)
        return {"sequence": seq, "reward": reward, "done": done, "info": info}