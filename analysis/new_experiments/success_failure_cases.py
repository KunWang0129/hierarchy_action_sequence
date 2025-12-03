"""Success and failure case analysis: comparing high vs low-success participants.

Compares exploration patterns between high-success and low-success participants,
focusing on 2-key sequence usage in INCORRECT trials (since correct trials
trivially use valid sequences by design).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.functions import (
    PLOT_STYLE,
    _create_summary_dataframe,
    _flatten_blocks,
    parse_action_sequence,
    parse_goal_stars,
    parse_trial_correctness,
    setup_plot_style,
)
from inference.star_making.assets_utils import StarMakingRules

setup_plot_style()
LOG = logging.getLogger(__name__)
INVALID_2_SEQ = {(0, 2), (2, 1), (1, 3), (3, 1)}


# ============================================================================
# Data Loading
# ============================================================================

def load_conditions(high_path: Path, low_path: Path) -> dict[str, list[dict]]:
    """Load participants from transfer_high and transfer_low conditions."""
    data = {}
    for label, path in [("high", high_path), ("low", low_path)]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data[label] = json.load(f).get("participants", [])
        LOG.info("Loaded %d participants from %s", len(data[label]), label)
    return data


# ============================================================================
# Participant Success Rate
# ============================================================================

def compute_participant_success_rates(
    correctness: dict[int, list[list[bool]]],
    phase: str = "all",
) -> dict[int, float]:
    """Compute success rate per participant."""
    rates = {}
    for pid, blocks in correctness.items():
        trials = _flatten_blocks(blocks)
        if trials:
            rates[pid] = sum(trials) / len(trials)
    return rates


def select_top_bottom_participants(
    rates: dict[int, float],
    top_n: int = 10,
) -> tuple[set[int], set[int]]:
    """Select top-N highest and bottom-N lowest success rate participants."""
    sorted_pids = sorted(rates.keys(), key=lambda p: rates[p], reverse=True)
    top = set(sorted_pids[:top_n])
    bottom = set(sorted_pids[-top_n:])
    return top, bottom


# ============================================================================
# Count 2-key Usage (Incorrect Trials Only)
# ============================================================================

def count_2seq_incorrect_trials(
    sequences: dict[int, list[list[list[int]]]],
    correctness: dict[int, list[list[bool]]],
    participant_group: dict[int, str],
    transfer_type: str,
    phase: str,
    num_blocks: int | None = None,
) -> list[dict]:
    """Count valid/invalid 2-key pairs in INCORRECT trials only."""
    rules = StarMakingRules()
    valid_pairs = set(rules.learning_rules["low"].keys())
    rows: list[dict] = []

    for pid, blocks in sequences.items():
        if pid not in participant_group:
            continue

        selected_blocks = blocks[:num_blocks] if num_blocks else blocks
        seqs = _flatten_blocks(selected_blocks)
        correct_blocks = correctness.get(pid, [])
        correct_blocks = correct_blocks[:num_blocks] if num_blocks else correct_blocks
        correct = _flatten_blocks(correct_blocks)
        n = min(len(seqs), len(correct))

        for idx in range(n):
            # Only analyze INCORRECT trials
            if correct[idx]:
                continue

            seq = seqs[idx]
            if len(seq) < 4:
                continue

            pairs = [(seq[0], seq[1]), (seq[2], seq[3])]
            valid_count = sum(1 for p in pairs if p in valid_pairs)
            invalid_count = sum(1 for p in pairs if p in INVALID_2_SEQ)

            rows.append({
                "participant_id": pid,
                "group": participant_group[pid],
                "trial_index": idx,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "transfer_type": transfer_type,
                "phase": phase,
            })
    return rows


# ============================================================================
# Experiment 4: Valid Pair Proportion in OLD Sequences
# ============================================================================

def build_learning_memory_for_participants(
    learning_sequences: dict[int, list[list[list[int]]]],
    learning_correctness: dict[int, list[list[bool]]],
    learning_goals: dict[int, list[list[str | None]]],
    participant_ids: set[int],
) -> dict[int, dict]:
    """Build learning memory for specified participants only.

    Captures rewarded 4-key sequences per star for each participant in the set.

    Args:
        learning_sequences: All learning sequences per participant
        learning_correctness: All learning correctness per participant
        learning_goals: All learning goals per participant
        participant_ids: Set of participant IDs to include

    Returns:
        Dict mapping participant_id to their learning memory:
        {
            participant_id: {
                "per_star": {star: set of 4-key tuples},
                "all_sequences": set of all 4-key tuples
            }
        }
    """
    memory: dict = {}

    for participant_id in participant_ids:
        if participant_id not in learning_sequences:
            continue

        seq_blocks = learning_sequences[participant_id]
        seqs = _flatten_blocks(seq_blocks)
        correctness = _flatten_blocks(learning_correctness.get(participant_id, []))
        goals = _flatten_blocks(learning_goals.get(participant_id, []))

        if not seqs or not correctness or not goals:
            LOG.debug(
                "Skipping participant %s due to missing learning data (seq:%s cor:%s goals:%s)",
                participant_id, len(seqs), len(correctness), len(goals),
            )
            continue

        n = min(len(seqs), len(correctness), len(goals))
        per_star: dict[str, set[tuple[int, ...]]] = defaultdict(set)
        all_sequences: set[tuple[int, ...]] = set()

        for idx in range(n):
            if not correctness[idx]:
                continue
            seq = seqs[idx]
            if len(seq) < 4:
                continue
            seq_tuple = tuple(seq[:4])
            star = goals[idx] or "Unknown"
            per_star[star].add(seq_tuple)
            all_sequences.add(seq_tuple)

        if per_star:
            memory[participant_id] = {"per_star": per_star, "all_sequences": all_sequences}

    LOG.info("Built learning memory for %s participants", len(memory))
    return memory


def calculate_valid_pair_proportion_for_groups(
    transfer_sequences: dict[int, list[list[list[int]]]],
    transfer_goals: dict[int, list[list[str | None]]],
    learning_memory: dict[int, dict],
    participant_group: dict[int, str],
    valid_pairs: set[tuple[int, int]],
    max_trials: int = 40,
) -> list[dict]:
    """Calculate per-trial proportion of valid pairs in OLD sequences by group.

    For each trial, computes the percentage of valid 2-key pairs that appear
    within 4-key sequences identical to rewarded sequences from learning.
    Trials with no valid pairs are excluded from the output.

    Args:
        transfer_sequences: Transfer phase sequences
        transfer_goals: Transfer phase goals
        learning_memory: Learning memory built by build_learning_memory_for_participants
        participant_group: Dict mapping participant_id -> "high" or "low"
        valid_pairs: Set of valid 2-key pairs
        max_trials: Maximum number of transfer trials to analyze

    Returns:
        List of row dicts with:
        - participant_id
        - group ("high" or "low")
        - trial_index
        - total_valid_pairs (count of valid pairs in this trial)
        - valid_pairs_in_old (count of valid pairs within OLD sequences)
        - percentage_in_old (percentage)
    """
    rows: list[dict] = []

    for participant_id, group in participant_group.items():
        if participant_id not in transfer_sequences:
            continue

        seq_blocks = transfer_sequences[participant_id]
        seqs = _flatten_blocks(seq_blocks)
        goals_blocks = transfer_goals.get(participant_id, [])
        goals = _flatten_blocks(goals_blocks)

        if not seqs:
            continue

        limit = min(len(seqs), max_trials)
        memo = learning_memory.get(participant_id, {"per_star": {}, "all_sequences": set()})
        all_sequences = memo.get("all_sequences", set())

        for idx in range(limit):
            seq = seqs[idx]
            if len(seq) < 4:
                continue

            seq_tuple = tuple(seq[:4])
            is_old = seq_tuple in all_sequences

            # Extract 2-key pairs and count valid ones
            pairs = [(seq[0], seq[1]), (seq[2], seq[3])]
            valid_count = sum(1 for pair in pairs if pair in valid_pairs)

            # Skip trials with no valid pairs (exclude from calculation)
            if valid_count == 0:
                continue

            # Count valid pairs that are in OLD sequences
            valid_in_old = valid_count if is_old else 0

            # Calculate percentage
            percentage = (valid_in_old / valid_count) * 100.0

            rows.append({
                "participant_id": participant_id,
                "group": group,
                "trial_index": idx + 1,
                "total_valid_pairs": valid_count,
                "valid_pairs_in_old": valid_in_old,
                "percentage_in_old": percentage,
            })

    LOG.info(
        "Computed proportion of valid pairs in OLD sequences for %s trials",
        len(rows),
    )
    return rows


def plot_valid_pair_proportion_by_group(
    rows: Sequence[dict],
    max_trials: int = 40,
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot proportion of valid pairs in OLD sequences by success group.

    Creates a line plot with:
    - X-axis: Transfer trial index
    - Y-axis: % of valid pairs in OLD sequences
    - Two lines: High-Success vs Low-Success groups
    - Error bars showing SEM
    """
    # Filter to max_trials
    filtered_rows = [r for r in rows if r["trial_index"] <= max_trials]

    fig, ax = plt.subplots(figsize=(10, 6))

    for group, label, color in [("high", "High-Success", "#2a9d8f"), ("low", "Low-Success", "#e63946")]:
        trials, means, sems = _aggregate_by_trial_group(
            filtered_rows, group, key="percentage_in_old", max_trial_index=max_trials
        )
        if trials:
            ax.errorbar(trials, means, yerr=sems, fmt="o-", color=color,
                        label=label, markersize=PLOT_STYLE["markersize"],
                        linewidth=PLOT_STYLE["linewidth"],
                        capsize=PLOT_STYLE["capsize"], alpha=0.8)

    ax.set_xlabel("Transfer Trial Index")
    ax.set_ylabel("% of Valid Pairs in OLD Sequences")
    ax.set_title("Transfer Phase: Valid Pair Reuse from OLD Sequences by Learning Success")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        LOG.info("Saved plot to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def fit_valid_pair_proportion_regression(rows: Sequence[dict]):
    """Fit model: percentage_in_old ~ group + trial + group:trial + (1|participant).

    Tests:
    - Main effect of group (do high-success participants have higher proportion?)
    - Trial effect (does proportion change over time?)
    - Group x Trial interaction (do groups show different trajectories?)

    Returns:
        (result, summary_dataframe)
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    data_rows = [dict(r) for r in rows]
    if not data_rows:
        raise ValueError("No data provided for regression")

    mean_trial = mean(r["trial_index"] for r in data_rows)
    for r in data_rows:
        r["trial_centered"] = r["trial_index"] - mean_trial
        r["group_num"] = 1 if r["group"] == "high" else 0

    df = pd.DataFrame(data_rows)
    formula = "percentage_in_old ~ group_num + trial_centered + group_num:trial_centered"

    # Try mixed model
    try:
        model = smf.mixedlm(formula, data=df, groups=df["participant_id"], re_formula="1")
        result = model.fit(reml=False, method="lbfgs", disp=False)
        return result, _create_summary_dataframe(result)
    except Exception as e:
        LOG.warning("Mixed model failed (%s); using clustered OLS", e)

    model = smf.ols(formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


def run_valid_pair_proportion_analysis(
    seqs: dict[str, dict[int, list[list[list[int]]]]],
    corr: dict[str, dict[int, list[list[bool]]]],
    goals: dict[str, dict[int, list[list[str | None]]]],
    condition_label: str,
    top_n: int,
    output_dir: Path,
    show: bool = False,
) -> list[dict]:
    """Run Experiment 4: Valid pair proportion in OLD sequences by learning success.

    Steps:
    1. Select top/bottom participants by LEARNING phase success
    2. Build learning memory for these participants
    3. Calculate valid pair proportions in transfer phase
    4. Plot comparison
    5. Run regression analysis

    Args:
        seqs: Parsed action sequences dict with 'learning_trials' and 'transfer_trials'
        corr: Parsed correctness dict with 'learning_trials' and 'transfer_trials'
        goals: Parsed goals dict with 'learning_trials' and 'transfer_trials'
        condition_label: Label for the experimental condition (e.g., 'high', 'low')
        top_n: Number of top/bottom participants
        output_dir: Directory for saving plots
        show: Whether to display plots interactively

    Returns:
        List of row dicts for analysis
    """
    # Select participants by learning phase success
    rates, top_pids, bottom_pids = select_by_phase_success(
        corr["learning_trials"], corr["transfer_trials"],
        phase="learning", top_n=top_n
    )

    LOG.info("Exp 4: Selecting by learning phase success")
    LOG.info("  Top %d (rates): %s", top_n,
             {p: f"{rates[p]:.2f}" for p in sorted(top_pids)})
    LOG.info("  Bottom %d (rates): %s", top_n,
             {p: f"{rates[p]:.2f}" for p in sorted(bottom_pids)})

    # Create participant group mapping
    participant_group = {}
    for pid in top_pids:
        participant_group[pid] = "high"
    for pid in bottom_pids:
        participant_group[pid] = "low"

    # Build learning memory for these participants
    all_participant_ids = top_pids | bottom_pids
    learning_memory = build_learning_memory_for_participants(
        seqs["learning_trials"],
        corr["learning_trials"],
        goals["learning_trials"],
        all_participant_ids
    )

    # Get valid pairs from rules
    rules = StarMakingRules()
    valid_pairs = set(rules.learning_rules["low"].keys())

    # Calculate proportions in transfer phase
    rows = calculate_valid_pair_proportion_for_groups(
        seqs["transfer_trials"],
        goals["transfer_trials"],
        learning_memory,
        participant_group,
        valid_pairs,
        max_trials=40
    )

    LOG.info("  Transfer phase: %d rows with valid pairs", len(rows))

    # Generate plot
    prefix = f"exp4_{condition_label}_learning_groups"
    plot_valid_pair_proportion_by_group(
        rows,
        max_trials=40,
        save_path=output_dir / f"{prefix}_valid_pair_proportion.png",
        show=show
    )

    # Fit regression model
    LOG.info("=" * 60)
    LOG.info("Exp 4 MODEL: Valid Pair Proportion (selected by learning success)")
    LOG.info("=" * 60)
    try:
        _, summary = fit_valid_pair_proportion_regression(rows)
        LOG.info("\n%s", summary.to_string(index=False))
    except Exception as e:
        LOG.warning("Model failed: %s", e)

    return rows


# ============================================================================
# Statistics Helper
# ============================================================================

def _aggregate_by_trial_group(
    rows: Sequence[dict],
    group: str,
    key: str = "valid_count",
    max_trial_index: int | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Compute mean and SEM per trial for a specific group.

    Args:
        rows: List of row dicts containing trial data
        group: Group to filter for ('high' or 'low')
        key: Data key to aggregate
        max_trial_index: If provided, extend indices to this max value
    """
    trial_data: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["group"] == group:
            trial_data[r["trial_index"]].append(float(r[key]))

    # Determine the range of indices to return
    if trial_data:
        data_max = max(trial_data.keys())
        if max_trial_index is not None:
            end_idx = max(data_max, max_trial_index)
        else:
            end_idx = data_max
        indices = list(range(min(trial_data.keys()), end_idx + 1))
    else:
        indices = []

    means, sems = [], []
    for t in indices:
        vals = trial_data.get(t, [])
        if vals:
            means.append(mean(vals))
            sems.append(0.0 if len(vals) <= 1 else pstdev(vals) / sqrt(len(vals)))
        else:
            # No data for this trial index - use NaN to create gap in line
            means.append(float("nan"))
            sems.append(float("nan"))
    return indices, means, sems


# ============================================================================
# Plot 1: Learning Curve by Group
# ============================================================================

def plot_learning_curve_by_group(
    rows: Sequence[dict],
    phase: str,
    save_path: Path | None = None,
    show: bool = False,
    max_trials: int = 40,
):
    """Plot valid 2-key usage over trials for high vs low-success groups."""
    # Filter to phase and limit to first max_trials indices
    phase_rows = [r for r in rows if r["phase"] == phase and r["trial_index"] < max_trials]

    fig, ax = plt.subplots(figsize=(10, 6))

    for group, label, color in [("high", "High-Success", "#2a9d8f"), ("low", "Low-Success", "#e63946")]:
        trials, means, sems = _aggregate_by_trial_group(
            phase_rows, group, max_trial_index=max_trials - 1
        )
        if trials:
            # No connecting line (fmt="o"), subtle error bars
            ax.errorbar(trials, means, yerr=sems, fmt="o", color=color,
                        label=label, markersize=PLOT_STYLE["markersize"],
                        capsize=2, elinewidth=0.8, alpha=0.7)

    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Mean Valid 2-Key Pairs (Incorrect Trials)")
    ax.set_title(f"{phase.capitalize()} Phase: Valid 2-Key Usage by Participant Group")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        LOG.info("Saved plot to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# Plot 2: Distribution Comparison (Box/Violin)
# ============================================================================

def plot_distribution_by_group(
    rows: Sequence[dict],
    phase: str,
    save_path: Path | None = None,
    show: bool = False,
):
    """Box plot comparing valid pair distribution between groups."""
    phase_rows = [r for r in rows if r["phase"] == phase]

    high_vals = [r["valid_count"] for r in phase_rows if r["group"] == "high"]
    low_vals = [r["valid_count"] for r in phase_rows if r["group"] == "low"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([low_vals, high_vals], labels=["Low-Success", "High-Success"],
                    patch_artist=True, widths=0.6)

    colors = ["#e63946", "#2a9d8f"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Valid 2-Key Pairs (per incorrect trial)")
    ax.set_title(f"{phase.capitalize()} Phase: Distribution by Participant Group")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        LOG.info("Saved plot to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# Plot 3: Participant Success Rate Distribution
# ============================================================================

def plot_success_rate_histogram(
    rates: dict[int, float],
    top_pids: set[int],
    bottom_pids: set[int],
    save_path: Path | None = None,
    show: bool = False,
):
    """Histogram of participant success rates with group markers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    all_rates = list(rates.values())
    ax.hist(all_rates, bins=15, color="#1d3557", alpha=0.7, edgecolor="black")

    # Mark top and bottom
    for pid in top_pids:
        ax.axvline(rates[pid], color="#2a9d8f", alpha=0.5, linewidth=1.5)
    for pid in bottom_pids:
        ax.axvline(rates[pid], color="#e63946", alpha=0.5, linewidth=1.5)

    ax.plot([], [], color="#2a9d8f", label="High-Success", linewidth=2)
    ax.plot([], [], color="#e63946", label="Low-Success", linewidth=2)

    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Count")
    ax.set_title("Participant Success Rate Distribution")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        LOG.info("Saved plot to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# Regression Model
# ============================================================================

def fit_group_comparison_model(rows: Sequence[dict], phase: str):
    """Fit model: valid_count ~ group + trial + group:trial + (1|participant)."""
    import pandas as pd
    import statsmodels.formula.api as smf

    phase_rows = [dict(r) for r in rows if r["phase"] == phase]
    if not phase_rows:
        raise ValueError(f"No {phase} data")

    mean_trial = mean(r["trial_index"] for r in phase_rows)
    for r in phase_rows:
        r["trial_centered"] = r["trial_index"] - mean_trial
        r["group_num"] = 1 if r["group"] == "high" else 0

    df = pd.DataFrame(phase_rows)
    formula = "valid_count ~ group_num + trial_centered + group_num:trial_centered"

    # Try mixed model
    try:
        model = smf.mixedlm(formula, data=df, groups=df["participant_id"], re_formula="1")
        result = model.fit(reml=False, method="lbfgs", disp=False)
        return result, _create_summary_dataframe(result)
    except Exception as e:
        LOG.warning("Mixed model failed (%s); using clustered OLS", e)

    model = smf.ols(formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


# ============================================================================
# Experiment 1 & 2: Phase-Specific Success Groups
# ============================================================================

def select_by_phase_success(
    correctness_learning: dict[int, list[list[bool]]],
    correctness_transfer: dict[int, list[list[bool]]],
    phase: str,
    top_n: int = 10,
) -> tuple[dict[int, float], set[int], set[int]]:
    """Select top/bottom participants by success rate in a specific phase.

    Args:
        correctness_learning: Learning phase correctness data per participant
        correctness_transfer: Transfer phase correctness data per participant
        phase: 'learning' or 'transfer' - which phase to use for selection
        top_n: Number of top and bottom participants to select

    Returns:
        (rates, top_pids, bottom_pids) where rates is success rate per participant
    """
    if phase == "learning":
        correctness = correctness_learning
    elif phase == "transfer":
        correctness = correctness_transfer
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 'learning' or 'transfer'")

    rates = compute_participant_success_rates(correctness)
    top_pids, bottom_pids = select_top_bottom_participants(rates, top_n)
    return rates, top_pids, bottom_pids


def run_phase_specific_analysis(
    seqs: dict[str, dict[int, list[list[list[int]]]]],
    corr: dict[str, dict[int, list[list[bool]]]],
    selection_phase: str,
    condition_label: str,
    top_n: int,
    output_dir: Path,
    show: bool = False,
) -> list[dict]:
    """Run analysis selecting participants by success in a specific phase.

    Args:
        seqs: Parsed action sequences dict with 'learning_trials' and 'transfer_trials'
        corr: Parsed correctness dict with 'learning_trials' and 'transfer_trials'
        selection_phase: 'learning' or 'transfer' - phase to use for selecting groups
        condition_label: Label for the experimental condition (e.g., 'high', 'low')
        top_n: Number of top/bottom participants
        output_dir: Directory for saving plots
        show: Whether to display plots interactively

    Returns:
        List of row dicts for analysis
    """
    exp_num = "1" if selection_phase == "learning" else "2"

    # Select participants by phase-specific success
    rates, top_pids, bottom_pids = select_by_phase_success(
        corr["learning_trials"], corr["transfer_trials"],
        phase=selection_phase, top_n=top_n
    )

    LOG.info("Exp %s: Selecting by %s phase success", exp_num, selection_phase)
    LOG.info("  Top %d (rates): %s", top_n,
             {p: f"{rates[p]:.2f}" for p in sorted(top_pids)})
    LOG.info("  Bottom %d (rates): %s", top_n,
             {p: f"{rates[p]:.2f}" for p in sorted(bottom_pids)})

    # Create participant group mapping
    participant_group = {}
    for pid in top_pids:
        participant_group[pid] = "high"
    for pid in bottom_pids:
        participant_group[pid] = "low"

    # Count 2-key usage in both phases for these participants
    all_rows: list[dict] = []
    for phase in ["learning", "transfer"]:
        rows = count_2seq_incorrect_trials(
            seqs[f"{phase}_trials"], corr[f"{phase}_trials"],
            participant_group, selection_phase, phase
        )
        all_rows.extend(rows)
        LOG.info("  %s phase: %d incorrect trial rows", phase, len(rows))

    # Generate plots
    prefix = f"exp{exp_num}_{condition_label}_{selection_phase}_groups"

    # Success rate histogram
    plot_success_rate_histogram(
        rates, top_pids, bottom_pids,
        output_dir / f"{prefix}_success_dist.png", show
    )

    # Learning curves and distributions for both phases
    for phase in ["learning", "transfer"]:
        plot_learning_curve_by_group(
            all_rows, phase,
            output_dir / f"{prefix}_{phase}_curve.png", show
        )
        plot_distribution_by_group(
            all_rows, phase,
            output_dir / f"{prefix}_{phase}_dist.png", show
        )

    # Fit models
    for phase in ["learning", "transfer"]:
        LOG.info("=" * 60)
        LOG.info("Exp %s MODEL: %s Phase (selected by %s success)",
                 exp_num, phase.upper(), selection_phase)
        LOG.info("=" * 60)
        try:
            _, summary = fit_group_comparison_model(all_rows, phase)
            LOG.info("\n%s", summary.to_string(index=False))
        except Exception as e:
            LOG.warning("Model failed: %s", e)

    return all_rows


# ============================================================================
# Experiment 3: Learning-Transfer Correlation
# ============================================================================

def compute_learning_transfer_correlation(
    correctness_learning: dict[int, list[list[bool]]],
    correctness_transfer: dict[int, list[list[bool]]],
) -> tuple[dict[int, float], dict[int, float], float, float, float, float]:
    """Compute correlation between learning and transfer phase success rates.

    Returns:
        (learning_rates, transfer_rates, pearson_r, pearson_p, spearman_r, spearman_p)
    """
    from scipy import stats

    learning_rates = compute_participant_success_rates(correctness_learning)
    transfer_rates = compute_participant_success_rates(correctness_transfer)

    # Get common participants
    common_pids = set(learning_rates.keys()) & set(transfer_rates.keys())
    if len(common_pids) < 3:
        raise ValueError("Not enough common participants for correlation")

    learning_vals = [learning_rates[pid] for pid in sorted(common_pids)]
    transfer_vals = [transfer_rates[pid] for pid in sorted(common_pids)]

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(learning_vals, transfer_vals)

    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(learning_vals, transfer_vals)

    return learning_rates, transfer_rates, pearson_r, pearson_p, spearman_r, spearman_p


def plot_learning_transfer_scatter(
    learning_rates: dict[int, float],
    transfer_rates: dict[int, float],
    pearson_r: float,
    pearson_p: float,
    spearman_r: float,
    spearman_p: float,
    condition_label: str = "",
    save_path: Path | None = None,
    show: bool = False,
):
    """Scatter plot of learning vs transfer success rates with regression line."""
    import numpy as np
    from scipy import stats

    common_pids = set(learning_rates.keys()) & set(transfer_rates.keys())
    x = [learning_rates[pid] for pid in sorted(common_pids)]
    y = [transfer_rates[pid] for pid in sorted(common_pids)]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(x, y, c="#1d3557", alpha=0.6, s=50, edgecolors="black", linewidths=0.5)

    # Regression line
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#e63946", linewidth=2, linestyle="--",
            label=f"y = {slope:.2f}x + {intercept:.2f}")

    # Add correlation annotation
    textstr = (f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})\n"
               f"Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.2e})\n"
               f"N = {len(common_pids)}")
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props)

    # Identity line (y=x) for reference
    lims = [min(min(x), min(y)), max(max(x), max(y))]
    ax.plot(lims, lims, "k:", alpha=0.5, label="y = x (identity)")

    ax.set_xlabel("Learning Phase Success Rate")
    ax.set_ylabel("Transfer Phase Success Rate")
    title = "Learning vs Transfer Success Rate"
    if condition_label:
        title += f" ({condition_label})"
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        LOG.info("Saved plot to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def run_learning_transfer_correlation_analysis(
    corr: dict[str, dict[int, list[list[bool]]]],
    condition_label: str,
    output_dir: Path,
    show: bool = False,
):
    """Run Experiment 3: Learning-Transfer correlation analysis."""
    LOG.info("=" * 60)
    LOG.info("Exp 3: Learning-Transfer Correlation (%s)", condition_label)
    LOG.info("=" * 60)

    learning_rates, transfer_rates, pr, pp, sr, sp = compute_learning_transfer_correlation(
        corr["learning_trials"], corr["transfer_trials"]
    )

    LOG.info("Pearson r = %.3f, p = %.2e", pr, pp)
    LOG.info("Spearman rho = %.3f, p = %.2e", sr, sp)

    # Interpretation
    if pp < 0.05:
        if pr > 0.5:
            LOG.info("Strong positive correlation: good learners tend to transfer well")
        elif pr > 0.3:
            LOG.info("Moderate positive correlation: some relationship between learning and transfer")
        elif pr > 0:
            LOG.info("Weak positive correlation")
        else:
            LOG.info("Negative correlation: unexpected pattern")
    else:
        LOG.info("No significant correlation (p >= 0.05)")

    plot_learning_transfer_scatter(
        learning_rates, transfer_rates, pr, pp, sr, sp,
        condition_label,
        output_dir / f"exp3_{condition_label}_learning_transfer_scatter.png",
        show
    )

    return pr, pp, sr, sp


# ============================================================================
# Main CLI
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Compare high vs low-success participants.")
    parser.add_argument("--transfer-high", type=Path, required=True)
    parser.add_argument("--transfer-low", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/plots/success_failure"))
    parser.add_argument("--top-n", type=int, default=10, help="Number of top/bottom participants")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_conditions(args.transfer_high, args.transfer_low)

    for cond, participants in data.items():
        seqs = parse_action_sequence(participants)
        corr = parse_trial_correctness(participants)
        goals = parse_goal_stars(participants)

        # Experiment 1: Select by LEARNING phase success
        LOG.info("\n" + "-" * 60)
        LOG.info("EXPERIMENT 1: %s condition - Learning Phase Selection", cond.upper())
        LOG.info("-" * 60)
        run_phase_specific_analysis(
            seqs, corr,
            selection_phase="learning",
            condition_label=cond,
            top_n=args.top_n,
            output_dir=args.output_dir,
            show=args.show
        )

        # Experiment 2: Select by TRANSFER phase success
        LOG.info("\n" + "-" * 60)
        LOG.info("EXPERIMENT 2: %s condition - Transfer Phase Selection", cond.upper())
        LOG.info("-" * 60)
        run_phase_specific_analysis(
            seqs, corr,
            selection_phase="transfer",
            condition_label=cond,
            top_n=args.top_n,
            output_dir=args.output_dir,
            show=args.show
        )

        # Experiment 3: Learning-Transfer Correlation
        LOG.info("\n" + "-" * 60)
        LOG.info("EXPERIMENT 3: %s condition - Learning-Transfer Correlation", cond.upper())
        LOG.info("-" * 60)
        run_learning_transfer_correlation_analysis(
            corr,
            condition_label=cond,
            output_dir=args.output_dir,
            show=args.show
        )

        # Experiment 4: Valid Pair Proportion in OLD Sequences by Learning Success
        LOG.info("\n" + "-" * 60)
        LOG.info("EXPERIMENT 4: %s condition - Valid Pair Proportion in OLD Sequences", cond.upper())
        LOG.info("-" * 60)
        run_valid_pair_proportion_analysis(
            seqs, corr, goals,
            condition_label=cond,
            top_n=args.top_n,
            output_dir=args.output_dir,
            show=args.show
        )

    LOG.info("\n" + "=" * 60)
    LOG.info("All experiments complete!")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()

"""
Sample usage:
    PYTHONPATH=. python analysis/new_experiments/success_failure_cases.py \
        --transfer-high results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json \
        --transfer-low results/gpt-oss-120b/transfer_low/item_20251122_040528_experiment.json \
        --output-dir results/gpt-oss-120b/experiments/success_failure \
        --top-n 10

    # With interactive display
    PYTHONPATH=. python analysis/new_experiments/success_failure_cases.py \
        --transfer-high results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json \
        --transfer-low results/gpt-oss-120b/transfer_low/item_20251122_040528_experiment.json \
        --output-dir results/gpt-oss-120b/experiments/success_failure \
        --top-n 10 --show
"""
