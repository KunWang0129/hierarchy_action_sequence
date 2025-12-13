"""Simple experiment loop for computational models."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Type, TypeVar

from inference.comp_model.star_making import StarMakingEnv

AgentT = TypeVar("AgentT")

LEARNING_GOALS = ["Star_0", "Star_1", "Star_2", "Star_3"] * 20
TRANSFER_GOALS = ["Star_1", "Star_2"] * 20


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _serialize_table(table: Dict[Any, float]) -> List[tuple[List[Any], float]]:
    return [(list(key) if isinstance(key, tuple) else [key], value) for key, value in table.items()]


def _snapshot_q(agent: Any) -> Dict[str, List[tuple[List[Any], float]]]:
    snapshot: Dict[str, List[tuple[List[Any], float]]] = {}
    if hasattr(agent, "q"):
        snapshot["q"] = _serialize_table(agent.q)  # type: ignore[arg-type]
    if hasattr(agent, "q_pair"):
        snapshot["q_pair"] = _serialize_table(agent.q_pair)  # type: ignore[arg-type]
    if hasattr(agent, "q_item"):
        snapshot["q_item"] = _serialize_table(agent.q_item)  # type: ignore[arg-type]
    return snapshot


def _run_phase(agent: Any, env: StarMakingEnv, goals: Sequence[str], rule_type: str) -> Dict[str, Any]:
    env.rule_type = rule_type
    trials: List[Dict[str, Any]] = []
    for goal in goals:
        result = agent.run_trial(env, goal_star=goal)
        result["goal_star"] = goal
        trials.append(result)
    rewards = [float(t["reward"]) for t in trials]
    return {"trials": trials, "avg_reward": _mean(rewards)}


def run_experiment(
    agent_cls: Type[AgentT],
    seeds: Iterable[int] | None = None,
    agent_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Run learning then transfer blocks for each seed and collect metrics."""
    agent_kwargs = agent_kwargs or {}
    seeds = list(range(120)) if seeds is None else list(seeds)
    results: List[Dict[str, Any]] = []

    for seed in seeds:
        env = StarMakingEnv(rule_type="learning", seed=seed)
        agent = agent_cls(seed=seed, **agent_kwargs)

        learning = _run_phase(agent, env, LEARNING_GOALS, "learning")
        learning["q_values_end"] = _snapshot_q(agent)

        env.rule_type = "transfer"
        transfer = _run_phase(agent, env, TRANSFER_GOALS, "transfer")
        transfer["q_values_end"] = _snapshot_q(agent)

        results.append(
            {
                "seed": seed,
                "learning": learning,
                "transfer": transfer,
            }
        )

    return results
