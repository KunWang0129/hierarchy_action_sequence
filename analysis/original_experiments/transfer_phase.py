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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from analysis.functions import (
    MixedRegressor,
    PLOT_COLORS,
    PLOT_MARKERS,
    PLOT_STYLE,
    _create_summary_dataframe,
    _flatten_blocks,
    parse_action_sequence,
    parse_trial_correctness,
    setup_plot_style,
)
from inference.star_making.assets_utils import StarMakingRules

setup_plot_style()

LOG = logging.getLogger(__name__)
INVALID_2_SEQ = {(0, 2), (2, 1), (1, 3), (3, 1)}


def _mean_sem(values: Sequence[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    m = mean(values)
    sem = pstdev(values) / sqrt(len(values))
    return m, sem


def _compute_time_series_stats(
    series: dict[int, list[int]], trial_order: Sequence[int]
) -> tuple[list[float], list[float]]:
    means, sems = [], []
    for t_idx in trial_order:
        vals = series.get(t_idx, [])
        m, sem = _mean_sem(vals)
        means.append(m)
        sems.append(sem)
    return means, sems


# ============================================================================
# Experiment 1: 2-Key Sequence Transfer
# ============================================================================


def detect_first_discovery_and_transfer(
    sequences: dict[int, list[list[list[int]]]],
    phase: str = "learning",
) -> list[dict]:
    """Track when 2-key sequences are discovered in one slot and transferred to the other.

    Args:
        sequences: Participant sequences from parse_action_sequence
        phase: "learning" or "transfer"

    Returns:
        List of dicts with participant_id, block, sequence, validity, trials_until_transfer
    """
    rules = StarMakingRules()
    valid_pairs = set(rules.learning_rules["low"].keys())

    rows: list[dict] = []

    for participant_id, blocks in sequences.items():
        for block_idx, block in enumerate(blocks):
            # Track first appearance of each 2-key sequence in each slot
            slot1_first: dict[tuple, int] = {}  # sequence -> trial_idx
            slot2_first: dict[tuple, int] = {}

            for trial_idx, seq in enumerate(block):
                if len(seq) < 4:
                    continue

                pair1 = tuple(seq[0:2])
                pair2 = tuple(seq[2:4])

                # Record first appearance in each slot
                if pair1 not in slot1_first:
                    slot1_first[pair1] = trial_idx
                if pair2 not in slot2_first:
                    slot2_first[pair2] = trial_idx

            # Now measure transfer: slot1 -> slot2 and slot2 -> slot1
            all_sequences = set(slot1_first.keys()) | set(slot2_first.keys())

            for seq in all_sequences:
                # Classify validity
                if seq in valid_pairs:
                    validity = "valid"
                elif seq in INVALID_2_SEQ:
                    validity = "invalid"
                else:
                    continue  # Skip sequences that are neither valid nor tracked invalid

                # Measure slot1 -> slot2 transfer
                if seq in slot1_first and seq in slot2_first:
                    first_in_1 = slot1_first[seq]
                    first_in_2 = slot2_first[seq]
                    if first_in_1 < first_in_2:
                        trials_until_transfer = first_in_2 - first_in_1
                        rows.append({
                            "participant_id": participant_id,
                            "block": block_idx,
                            "sequence": seq,
                            "validity": validity,
                            "trials_until_transfer": trials_until_transfer,
                            "phase": phase,
                        })

    return rows


def compute_transfer_difference(rows: Sequence[dict]) -> list[dict]:
    """Compute per-participant, per-block difference: (valid - invalid) transfer times.

    Args:
        rows: Output from detect_first_discovery_and_transfer

    Returns:
        List of dicts with participant_id, block, phase, difference
    """
    # Group by participant, block, phase
    groups: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: {"valid": [], "invalid": []})

    for row in rows:
        key = (row["participant_id"], row["block"], row["phase"])
        groups[key][row["validity"]].append(row["trials_until_transfer"])

    # Compute differences
    differences: list[dict] = []
    for (pid, block, phase), times in groups.items():
        valid_mean = mean(times["valid"]) if times["valid"] else None
        invalid_mean = mean(times["invalid"]) if times["invalid"] else None

        if valid_mean is not None and invalid_mean is not None:
            differences.append({
                "participant_id": pid,
                "block": block,
                "phase": phase,
                "difference": valid_mean - invalid_mean,
                "valid_mean": valid_mean,
                "invalid_mean": invalid_mean,
            })

    return differences


def fit_transfer_difference_regression(rows: Sequence[dict]):
    """Run mixed-effects regression on transfer time difference.

    Formula: difference ~ block + (1|participant_id)
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    # Center block
    mean_block = mean(r["block"] for r in rows)
    for r in rows:
        r["block_centered"] = r["block"] - mean_block

    reg = MixedRegressor(rows, participant_col="participant_id")

    # Try mixed model
    try:
        return reg.fit("difference", ["block_centered"])
    except Exception as exc:
        LOG.warning("Mixed model failed (%s); retrying with powell", exc)

    # Try alternative optimizer
    try:
        return reg.fit("difference", ["block_centered"], method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed (%s); falling back to OLS", exc2)

    # Fallback to clustered OLS
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    model = smf.ols("difference ~ block_centered", data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


def _summarize_transfer_dataset(rows: Sequence[dict]) -> dict[str, tuple[float, float]]:
    summary = {"valid": [], "invalid": []}
    for row in rows:
        validity = row.get("validity")
        if validity in summary:
            summary[validity].append(int(row["trials_until_transfer"]))
    return {label: _mean_sem(values) for label, values in summary.items()}


def plot_transfer_times_by_phase(
    learning_high: Sequence[dict],
    learning_low: Sequence[dict],
    transfer_high: Sequence[dict],
    transfer_low: Sequence[dict],
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot 2x2 grid of bar plots showing valid vs invalid transfer times."""
    phases = [("Learning", [learning_high, learning_low]), ("Transfer", [transfer_high, transfer_low])]
    conditions = ["High", "Low"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    max_height = 0.0

    for row_idx, (phase_label, datasets) in enumerate(phases):
        for col_idx, condition_label in enumerate(conditions):
            data = datasets[col_idx]
            ax = axes[row_idx, col_idx]
            ax.set_title(f"{phase_label} – {condition_label}")

            if not data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_xticks([])
                continue

            stats = _summarize_transfer_dataset(data)
            heights = [stats["invalid"][0], stats["valid"][0]]
            errors = [stats["invalid"][1], stats["valid"][1]]
            max_height = max(max_height, *heights)

            bars = ax.bar([0, 1], heights, yerr=errors,
                          color=[PLOT_COLORS["invalid"], PLOT_COLORS["valid"]],
                          alpha=PLOT_STYLE["alpha"], capsize=PLOT_STYLE["capsize"], width=0.5)

            for bar, height in zip(bars, heights):
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                            f"{height:.1f}", ha="center", va="bottom", fontsize=10)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Invalid", "Valid"])
            if col_idx == 0:
                ax.set_ylabel("Trials until transfer")

    for axis in axes.flat:
        axis.set_ylim(0, max(max_height * 1.2, 1.0))

    fig.suptitle("2-Key Sequence Transfer Times", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig, axes


# ============================================================================
# Experiment 2: Transfer Accuracy Comparison
# ============================================================================


def extract_transfer_accuracy(
    participants_high: Sequence[dict],
    participants_low: Sequence[dict],
) -> list[dict]:
    """Extract trial-by-trial accuracy during transfer phase for both conditions.

    Args:
        participants_high: Participants from transfer_high experiment
        participants_low: Participants from transfer_low experiment

    Returns:
        Long-form list with participant_id, transfer_type, trial_index, accuracy
    """
    rows: list[dict] = []

    # Process transfer_high
    correctness_high = parse_trial_correctness(participants_high)["transfer_trials"]
    for participant_id, blocks in correctness_high.items():
        flat_trials = _flatten_blocks(blocks)
        for trial_idx, accuracy in enumerate(flat_trials):
            rows.append({
                "participant_id": participant_id,
                "transfer_type": "high",
                "trial_index": trial_idx,
                "accuracy": int(accuracy),
            })

    # Process transfer_low
    correctness_low = parse_trial_correctness(participants_low)["transfer_trials"]
    for participant_id, blocks in correctness_low.items():
        flat_trials = _flatten_blocks(blocks)
        for trial_idx, accuracy in enumerate(flat_trials):
            rows.append({
                "participant_id": participant_id,
                "transfer_type": "low",
                "trial_index": trial_idx,
                "accuracy": int(accuracy),
            })

    return rows


def fit_transfer_accuracy_regression(rows: Sequence[dict]):
    """Run mixed-effects regression on transfer accuracy.

    Formula: accuracy ~ transfer_type + trial_centered + transfer_type:trial_centered + (1|participant_id)
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    # Center trial index
    mean_trial = mean(r["trial_index"] for r in rows)
    for r in rows:
        r["trial_centered"] = r["trial_index"] - mean_trial
        r["transfer_type_num"] = 1 if r["transfer_type"] == "high" else 0

    reg = MixedRegressor(rows, participant_col="participant_id")
    formula_terms = ["transfer_type_num", "trial_centered", "transfer_type_num:trial_centered"]

    # Try mixed model
    try:
        return reg.fit("accuracy", formula_terms)
    except Exception as exc:
        LOG.warning("Mixed model failed (%s); retrying with powell", exc)

    # Try alternative optimizer
    try:
        return reg.fit("accuracy", formula_terms, method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed (%s); falling back to OLS", exc2)

    # Fallback to clustered OLS
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    model = smf.ols("accuracy ~ transfer_type_num + trial_centered + transfer_type_num:trial_centered", data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


def plot_transfer_accuracy_over_time(
    rows: Sequence[dict],
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot accuracy over transfer trials for high vs low conditions.

    Styled to match human experiment manipulation comparison plot.
    """
    if not rows:
        LOG.warning("No data to plot")
        return None, None

    # Group data by transfer type
    grouped = {"high": defaultdict(list), "low": defaultdict(list)}
    for row in rows:
        grouped[row["transfer_type"]][int(row["trial_index"])].append(int(row["accuracy"]))

    trial_indices = sorted({idx for series in grouped.values() for idx in series})
    stats = {label: _compute_time_series_stats(series, trial_indices)
             for label, series in grouped.items()}

    # Create figure with portrait orientation to match human experiment
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot error bars only (remove fill_between from original)
    for label in ("high", "low"):
        means, sems = stats[label]
        ax.errorbar(trial_indices, means, yerr=sems,
                    fmt='-o',
                    color=PLOT_COLORS[label],  # Uses updated global colors
                    linewidth=1.5,
                    markersize=6,
                    capsize=0,
                    elinewidth=1.5,
                    label=label)

    # Set axis labels with bold formatting (20pt)
    ax.set_xlabel("Trial", fontsize=20, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=20, fontweight='bold')

    # Set limits to match human experiment range
    if trial_indices:
        ax.set_xlim(trial_indices[0] - 0.5, trial_indices[-1] + 0.5)
    ax.set_ylim(0.0, 0.3)
    ax.set_yticks([0.1, 0.2, 0.3])
    ax.grid(False)

    # Format tick labels (16pt bold)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')

    # Explicitly remove spines (ensures consistency with human experiment)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Configure legend with title (matching human experiment)
    legend = ax.legend(title='Manipulation',
                      title_fontsize=18,
                      fontsize=16,
                      frameon=False,
                      loc='upper right')
    plt.setp(legend.get_title(), fontweight='bold')
    for text in legend.get_texts():
        text.set_fontweight('bold')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig, ax


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entrypoint for transfer phase analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Analyze transfer phase experiments.")
    parser.add_argument("--high_path", required=True, help="Path to transfer_high experiment.json")
    parser.add_argument("--low_path", required=True, help="Path to transfer_low experiment.json")
    parser.add_argument("--experiment", choices=["1", "2", "all"], default="all",
                        help="Which experiment to run")
    parser.add_argument("--output_dir", default="analysis/plots", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    # Load data
    high_path = Path(args.high_path)
    low_path = Path(args.low_path)

    if not high_path.exists():
        raise FileNotFoundError(f"High path not found: {high_path}")
    if not low_path.exists():
        raise FileNotFoundError(f"Low path not found: {low_path}")

    with open(high_path, "r", encoding="utf-8") as f:
        data_high = json.load(f)
    with open(low_path, "r", encoding="utf-8") as f:
        data_low = json.load(f)

    participants_high = data_high.get("participants", [])
    participants_low = data_low.get("participants", [])

    LOG.info("Loaded %s participants from transfer_high", len(participants_high))
    LOG.info("Loaded %s participants from transfer_low", len(participants_low))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Experiment 1: 2-Key Sequence Transfer
    # ========================================================================
    if args.experiment in ["1", "all"]:
        LOG.info("=" * 60)
        LOG.info("EXPERIMENT 1: 2-Key Sequence Transfer")
        LOG.info("=" * 60)

        # Parse sequences
        seq_high = parse_action_sequence(participants_high)
        seq_low = parse_action_sequence(participants_low)

        # Detect transfers
        learning_high_rows = detect_first_discovery_and_transfer(seq_high["learning_trials"], "learning")
        transfer_high_rows = detect_first_discovery_and_transfer(seq_high["transfer_trials"], "transfer")
        learning_low_rows = detect_first_discovery_and_transfer(seq_low["learning_trials"], "learning")
        transfer_low_rows = detect_first_discovery_and_transfer(seq_low["transfer_trials"], "transfer")

        LOG.info("Learning high transfer events: %s", len(learning_high_rows))
        LOG.info("Transfer high transfer events: %s", len(transfer_high_rows))
        LOG.info("Learning low transfer events: %s", len(learning_low_rows))
        LOG.info("Transfer low transfer events: %s", len(transfer_low_rows))

        # Plot transfer times
        plot_path = output_dir / "exp1_transfer_times.png"
        plot_transfer_times_by_phase(
            learning_high_rows, learning_low_rows,
            transfer_high_rows, transfer_low_rows,
            save_path=plot_path, show=args.show
        )
        LOG.info("Saved plot to %s", plot_path)

        # Compute differences and fit regression
        all_rows = learning_high_rows + transfer_high_rows + learning_low_rows + transfer_low_rows
        differences = compute_transfer_difference(all_rows)
        LOG.info("Computed %s difference rows", len(differences))

        if differences:
            try:
                result, summary = fit_transfer_difference_regression(differences)
                LOG.info("Transfer difference regression complete")
                LOG.info("\n%s", summary.to_string(index=False))
            except Exception as exc:
                LOG.warning("Regression failed: %s", exc)

    # ========================================================================
    # Experiment 2: Transfer Accuracy Comparison
    # ========================================================================
    if args.experiment in ["2", "all"]:
        LOG.info("=" * 60)
        LOG.info("EXPERIMENT 2: Transfer Accuracy Comparison")
        LOG.info("=" * 60)

        # Extract accuracy
        accuracy_rows = extract_transfer_accuracy(participants_high, participants_low)
        LOG.info("Extracted %s accuracy rows", len(accuracy_rows))

        # Plot accuracy over time
        plot_path = output_dir / "exp2_transfer_accuracy.png"
        plot_transfer_accuracy_over_time(accuracy_rows, save_path=plot_path, show=args.show)
        LOG.info("Saved plot to %s", plot_path)

        # Fit regression
        if accuracy_rows:
            try:
                result, summary = fit_transfer_accuracy_regression(accuracy_rows)
                LOG.info("Transfer accuracy regression complete")
                LOG.info("\n%s", summary.to_string(index=False))
            except Exception as exc:
                LOG.warning("Regression failed: %s", exc)


if __name__ == "__main__":
    main()

"""
Sample usage:
    # Run for item condition (comparing high vs low)
    python analysis/original_experiments/transfer_phase.py \
        --high_path results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json \
        --low_path results/gpt-oss-120b/transfer_low/item_20251122_040528_experiment.json \
        --experiment all \
        --output_dir results/gpt-oss-120b/experiments/transfer_phase/item

    # Run for no_item condition (comparing high vs low)
    python analysis/original_experiments/transfer_phase.py \
        --high_path results/gpt-oss-120b/transfer_high/no_item_20251122_075552_experiment.json \
        --low_path results/gpt-oss-120b/transfer_low/no_item_20251122_040337_experiment.json \
        --experiment all \
        --output_dir results/gpt-oss-120b/experiments/transfer_phase/no_item

    # Run specific experiment only
    python analysis/original_experiments/transfer_phase.py --high_path H.json --low_path L.json --experiment 1 --output_dir out/
    python analysis/original_experiments/transfer_phase.py --high_path H.json --low_path L.json --experiment 2 --output_dir out/ --show
"""
