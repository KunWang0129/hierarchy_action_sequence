"""Parameter sweep for computational models."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

from inference.comp_model.experiment import run_experiment
from inference.comp_model.q_learning import FlatQAgent, HierarchicalQAgent


def _compute_accuracy(results: List[Dict[str, Any]]) -> float:
    """Compute mean accuracy across all participants and trials."""
    total_trials = 0
    total_correct = 0

    for result in results:
        # Learning phase trials
        for trial in result["learning"]["trials"]:
            total_trials += 1
            if trial["reward"] > 0:
                total_correct += 1

        # Transfer phase trials
        for trial in result["transfer"]["trials"]:
            total_trials += 1
            if trial["reward"] > 0:
                total_correct += 1

    return total_correct / total_trials if total_trials > 0 else 0.0


def sweep_flat_agent(
    alpha_values: List[float] | None = None,
    beta_values: List[float] | None = None,
    n_seeds: int = 120,
) -> Tuple[Dict[str, Any], float]:
    """
    Perform parameter sweep for FlatQAgent.

    Args:
        alpha_values: Learning rates to test
        beta_values: Inverse temperatures to test
        n_seeds: Number of participants/seeds

    Returns:
        Tuple of (best_params, best_accuracy)
    """
    # Default parameter grids
    if alpha_values is None:
        alpha_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    if beta_values is None:
        beta_values = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]

    best_accuracy = 0.0
    best_params: Dict[str, Any] = {}

    total_combinations = len(alpha_values) * len(beta_values)
    current = 0

    print(f"Starting FlatQAgent sweep with {total_combinations} parameter combinations...")
    print(f"Alpha values: {alpha_values}")
    print(f"Beta values: {beta_values}")
    print(f"Number of seeds: {n_seeds}\n")

    for alpha, beta in itertools.product(alpha_values, beta_values):
        current += 1
        agent_kwargs = {"alpha": alpha, "beta": beta}

        results = run_experiment(
            agent_cls=FlatQAgent,
            seeds=range(n_seeds),
            agent_kwargs=agent_kwargs,
        )

        accuracy = _compute_accuracy(results)

        print(f"[{current}/{total_combinations}] alpha={alpha:.2f}, beta={beta:>4.1f} -> accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = agent_kwargs

    return best_params, best_accuracy


def sweep_hierarchical_agent(
    alpha_pair_values: List[float] | None = None,
    alpha_item_values: List[float] | None = None,
    beta_pair_values: List[float] | None = None,
    lambda_values: List[float] | None = None,
    n_seeds: int = 120,
) -> Tuple[Dict[str, Any], float]:
    """
    Perform parameter sweep for HierarchicalQAgent.

    Args:
        alpha_pair_values: Low-level learning rates to test
        alpha_item_values: High-level learning rates to test
        beta_pair_values: Inverse temperatures to test
        lambda_values: Item-level value weights to test
        n_seeds: Number of participants/seeds

    Returns:
        Tuple of (best_params, best_accuracy)
    """
    # Default parameter grids
    if alpha_pair_values is None:
        alpha_pair_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if alpha_item_values is None:
        alpha_item_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if beta_pair_values is None:
        beta_pair_values = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75]
    if lambda_values is None:
        lambda_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    best_accuracy = 0.0
    best_params: Dict[str, Any] = {}

    total_combinations = (
        len(alpha_pair_values)
        * len(alpha_item_values)
        * len(beta_pair_values)
        * len(lambda_values)
    )
    current = 0

    print(f"Starting HierarchicalQAgent sweep with {total_combinations} parameter combinations...")
    print(f"Alpha_pair values: {alpha_pair_values}")
    print(f"Alpha_item values: {alpha_item_values}")
    print(f"Beta_pair values: {beta_pair_values}")
    print(f"Lambda values: {lambda_values}")
    print(f"Number of seeds: {n_seeds}\n")

    for alpha_pair, alpha_item, beta_pair, lambda_ in itertools.product(
        alpha_pair_values, alpha_item_values, beta_pair_values, lambda_values
    ):
        current += 1
        agent_kwargs = {
            "alpha_pair": alpha_pair,
            "alpha_item": alpha_item,
            "beta_pair": beta_pair,
            "lambda_": lambda_,
        }

        results = run_experiment(
            agent_cls=HierarchicalQAgent,
            seeds=range(n_seeds),
            agent_kwargs=agent_kwargs,
        )

        accuracy = _compute_accuracy(results)

        print(
            f"[{current}/{total_combinations}] "
            f"alpha_pair={alpha_pair:.2f}, alpha_item={alpha_item:.2f}, "
            f"beta_pair={beta_pair:>4.1f}, lambda={lambda_:.2f} -> accuracy={accuracy:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = agent_kwargs

    return best_params, best_accuracy


def main() -> None:
    """Run parameter sweeps for both agents."""
    print("=" * 80)
    print("FLAT Q-LEARNING AGENT PARAMETER SWEEP")
    print("=" * 80)
    flat_params, flat_accuracy = sweep_flat_agent()
    print(f"\n{'=' * 80}")
    print(f"BEST FLAT AGENT PARAMETERS:")
    print(f"  Accuracy: {flat_accuracy:.4f}")
    for key, value in flat_params.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 80}\n\n")

    print("=" * 80)
    print("HIERARCHICAL Q-LEARNING AGENT PARAMETER SWEEP")
    print("=" * 80)
    hier_params, hier_accuracy = sweep_hierarchical_agent()
    print(f"\n{'=' * 80}")
    print(f"BEST HIERARCHICAL AGENT PARAMETERS:")
    print(f"  Accuracy: {hier_accuracy:.4f}")
    for key, value in hier_params.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 80}\n\n")

    # Print final summary of both agents
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - BEST PARAMETERS FOR BOTH AGENTS")
    print("=" * 80)
    print("\nFLAT Q-LEARNING AGENT:")
    print(f"  Accuracy: {flat_accuracy:.4f}")
    for key, value in flat_params.items():
        print(f"  {key}: {value}")

    print("\nHIERARCHICAL Q-LEARNING AGENT:")
    print(f"  Accuracy: {hier_accuracy:.4f}")
    for key, value in hier_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
