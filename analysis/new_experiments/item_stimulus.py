"""
Implementation of analysis for the following experiment:
Files to load: 4 paths:
- original transfer high
- original transfer low
- no item transfer high
- no item transfer low

# Experiment 1: Learning and transferring with 2-key sequences
Question: Do appearing items improve chunking of 2-key sequences during learning and  reuse of these chunks during high vs low transfer?
Design:
In each trial, count number of valid and invalid 2-key sequences
Use incorrect trials from the first 2 blocks for both learning and transfer
Predictors:
Sequence validity (V)
Trial numbers (T)
ItemFeedback (F)
TransferType (L)
Plot:
Plot content: number of 2-key sequences over trials
Red line: invalid, blue line: valid
First 20 trials
One plot with 2 subplots: learning phase (use transfer low for both)
One plot with 4 subplots: transfer phase for both item and no item
Model: mixed-effect regressor predicting number of sequences
Regressor 1: Learning only model
Ypmtv=β_0+β_1V+β_2T+β_3F+β_4(VT)+β_5(VF)+β_6(VTF)+u0p+ε.
Regressor 2: Transfer model
Ypmtv=β_0+β_1V+β_2T+β_3F+β_4L+β_5(VTt)+β_6(VTF)+β_7(VTL)+β_8(VTFL)+u0p+ε.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
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
ACCURACY_HIGH_COLOR = "#E15759"  # red, matches manipulation high in Section 2 styling
ACCURACY_LOW_COLOR = "#4EBFD8"   # cyan, matches manipulation low in Section 2 styling


# ============================================================================
# Data Loading
# ============================================================================

def load_four_conditions(
    original_high_path: Path,
    original_low_path: Path,
    no_item_high_path: Path,
    no_item_low_path: Path,
) -> dict[str, list[dict]]:
    """Load participants from all four experimental conditions."""
    paths = {
        "original_high": original_high_path,
        "original_low": original_low_path,
        "no_item_high": no_item_high_path,
        "no_item_low": no_item_low_path,
    }

    data = {}
    for condition, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{condition} file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        data[condition] = json_data.get("participants", [])
        LOG.info("Loaded %s participants from %s", len(data[condition]), condition)

    return data


# ============================================================================
# Counting Function
# ============================================================================

def count_valid_invalid_2seq_blocks(
    sequences: dict[int, list[list[list[int]]]],
    correctness: dict[int, list[list[bool]]],
    item_feedback: int,  # 1 = original, 0 = no_item
    transfer_type: str,  # "high" or "low"
    phase: str,          # "learning" or "transfer"
    num_blocks: int = 2,
) -> list[dict]:
    """Count valid/invalid 2-key sequences from first num_blocks blocks (incorrect trials only).

    Args:
        sequences: Participant sequences from parse_action_sequence
        correctness: Trial correctness from parse_trial_correctness
        item_feedback: 1 if original (with items), 0 if no_item
        transfer_type: "high" or "low"
        phase: "learning" or "transfer"
        num_blocks: Number of blocks to analyze (default 2 = 20 trials)

    Returns:
        Long-form list with separate rows for valid and invalid counts
    """
    rules = StarMakingRules()
    valid_pairs = set(rules.learning_rules["low"].keys())

    long_rows: list[dict] = []

    for participant_id, blocks in sequences.items():
        # Take first num_blocks blocks
        selected_blocks = blocks[:num_blocks]
        seqs = _flatten_blocks(selected_blocks)
        correct_blocks = correctness.get(participant_id, [])[:num_blocks]
        correct = _flatten_blocks(correct_blocks)

        if len(seqs) != len(correct):
            LOG.warning(
                "Sequence/correctness mismatch for participant %s (%s vs %s)",
                participant_id, len(seqs), len(correct)
            )

        n = min(len(seqs), len(correct))
        for trial_idx in range(n):
            # Only count incorrect trials
            if correct[trial_idx]:
                continue

            seq = seqs[trial_idx]
            if len(seq) < 4:
                continue

            # Count valid and invalid pairs
            pairs = [(seq[0], seq[1]), (seq[2], seq[3])]
            valid_count = sum(1 for pair in pairs if pair in valid_pairs)
            invalid_count = sum(1 for pair in pairs if pair in INVALID_2_SEQ)

            # Create long-form rows
            long_rows.append({
                "participant_id": participant_id,
                "trial_index": trial_idx,
                "validity": 1,
                "count": valid_count,
                "item_feedback": item_feedback,
                "transfer_type": transfer_type,
                "phase": phase,
            })
            long_rows.append({
                "participant_id": participant_id,
                "trial_index": trial_idx,
                "validity": 0,
                "count": invalid_count,
                "item_feedback": item_feedback,
                "transfer_type": transfer_type,
                "phase": phase,
            })

    return long_rows


# ============================================================================
# Statistics Helper
# ============================================================================

def _compute_trial_statistics(data_dict: dict[int, list[int]]) -> tuple[list[float], list[float]]:
    """Compute means and SEMs for trial data."""
    means, sems = [], []
    for t_idx in sorted(data_dict.keys()):
        vals = data_dict.get(t_idx, [])
        means.append(mean(vals) if vals else 0.0)
        sems.append(0.0 if len(vals) <= 1 else pstdev(vals) / sqrt(len(vals)))
    return means, sems


# ============================================================================
# Plot 1: Learning Phase Comparison
# ============================================================================

def plot_learning_phase_comparison(
    original_rows: Sequence[dict],
    no_item_rows: Sequence[dict],
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot learning phase: 2 subplots comparing original vs no_item."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = [
        (original_rows, "Original (with items)", axes[0]),
        (no_item_rows, "No Item", axes[1]),
    ]

    for rows, title, ax in datasets:
        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title, fontweight="bold")
            continue

        # Group by validity and trial
        valid_counts: dict[int, list[int]] = defaultdict(list)
        invalid_counts: dict[int, list[int]] = defaultdict(list)

        for row in rows:
            t_idx = int(row["trial_index"])
            count = int(row["count"])
            if row["validity"] == 1:
                valid_counts[t_idx].append(count)
            else:
                invalid_counts[t_idx].append(count)

        trial_indices = sorted(set(valid_counts) | set(invalid_counts))
        valid_means, valid_sems = _compute_trial_statistics(valid_counts)
        invalid_means, invalid_sems = _compute_trial_statistics(invalid_counts)

        for label, means, sems in [("valid", valid_means, valid_sems), ("invalid", invalid_means, invalid_sems)]:
            ax.errorbar(trial_indices, means, yerr=sems,
                        fmt=f"{PLOT_MARKERS[label]}-", color=PLOT_COLORS[label],
                        label=label.capitalize(), linewidth=PLOT_STYLE["linewidth"],
                        markersize=PLOT_STYLE["markersize"], capsize=PLOT_STYLE["capsize"])

        ax.set_xlabel("Trial Index")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(frameon=False)

    fig.suptitle("Learning Phase: Valid vs Invalid 2-Key Sequences (First 20 Trials)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig, axes


# ============================================================================
# Plot 2: Transfer Phase All Conditions
# ============================================================================

def plot_transfer_phase_all_conditions(
    original_high_rows: Sequence[dict],
    original_low_rows: Sequence[dict],
    no_item_high_rows: Sequence[dict],
    no_item_low_rows: Sequence[dict],
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot transfer phase: 4 subplots (2x2) for all condition combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    datasets = [
        (original_high_rows, "Original - Transfer High", (0, 0)),
        (original_low_rows, "Original - Transfer Low", (0, 1)),
        (no_item_high_rows, "No Item - Transfer High", (1, 0)),
        (no_item_low_rows, "No Item - Transfer Low", (1, 1)),
    ]

    for rows, title, (row_idx, col_idx) in datasets:
        ax = axes[row_idx, col_idx]

        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title, fontweight="bold")
            continue

        # Group by validity and trial
        valid_counts: dict[int, list[int]] = defaultdict(list)
        invalid_counts: dict[int, list[int]] = defaultdict(list)

        for row in rows:
            t_idx = int(row["trial_index"])
            count = int(row["count"])
            if row["validity"] == 1:
                valid_counts[t_idx].append(count)
            else:
                invalid_counts[t_idx].append(count)

        trial_indices = sorted(set(valid_counts) | set(invalid_counts))
        valid_means, valid_sems = _compute_trial_statistics(valid_counts)
        invalid_means, invalid_sems = _compute_trial_statistics(invalid_counts)

        for label, means, sems in [("valid", valid_means, valid_sems), ("invalid", invalid_means, invalid_sems)]:
            ax.errorbar(trial_indices, means, yerr=sems,
                        fmt=f"{PLOT_MARKERS[label]}-", color=PLOT_COLORS[label],
                        label=label.capitalize(), linewidth=PLOT_STYLE["linewidth"],
                        markersize=PLOT_STYLE["markersize"], capsize=PLOT_STYLE["capsize"])

        ax.set_xlabel("Trial Index")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(frameon=False)

    fig.suptitle("Transfer Phase: Valid vs Invalid 2-Key Sequences (First 20 Trials)",
                 fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig, axes


# ============================================================================
# Regression Model 1: Learning Only
# ============================================================================

def fit_learning_model(rows: Sequence[dict]):
    """Run mixed-effects regression on learning phase data.

    Formula: count ~ V + T + F + V:T + V:F + V:T:F + (1|participant)

    Where:
    - V = validity (0=invalid, 1=valid)
    - T = trial_centered
    - F = item_feedback (0=no_item, 1=original)
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    # Filter to learning phase only
    learning_rows = [r for r in rows if r.get("phase") == "learning"]
    if not learning_rows:
        raise ValueError("No learning phase data found")

    # Center trial index
    mean_trial = mean(r["trial_index"] for r in learning_rows)
    for r in learning_rows:
        r["trial_centered"] = r["trial_index"] - mean_trial

    reg = MixedRegressor(learning_rows, participant_col="participant_id")
    formula_terms = [
        "validity", "trial_centered", "item_feedback",
        "validity:trial_centered", "validity:item_feedback",
        "validity:trial_centered:item_feedback"
    ]

    # Try mixed model with primary optimizer
    try:
        return reg.fit("count", formula_terms)
    except Exception as exc:
        LOG.warning("Mixed model failed (%s); retrying with powell", exc)

    # Try alternative optimizer
    try:
        return reg.fit("count", formula_terms, method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed (%s); falling back to OLS", exc2)

    # Fallback to clustered OLS
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(learning_rows)
    formula = "count ~ validity + trial_centered + item_feedback + validity:trial_centered + validity:item_feedback + validity:trial_centered:item_feedback"
    model = smf.ols(formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


# ============================================================================
# Accuracy Data Collection
# ============================================================================

def collect_accuracy_rows(
    correctness: dict[int, list[list[bool]]],
    item_feedback: int,
    transfer_type: str,
    phase: str,
    num_blocks: int = 2,
) -> list[dict]:
    """Collect per-trial accuracy data in long form.

    Args:
        correctness: Trial correctness from parse_trial_correctness
        item_feedback: 1 if original (with items), 0 if no_item
        transfer_type: "high" or "low"
        phase: "learning" or "transfer"
        num_blocks: Number of blocks to analyze (default 2 = 20 trials)

    Returns:
        Long-form list with one row per trial per participant
    """
    rows: list[dict] = []

    for participant_id, blocks in correctness.items():
        selected_blocks = blocks[:num_blocks]
        correct = _flatten_blocks(selected_blocks)

        for trial_idx, is_correct in enumerate(correct):
            rows.append({
                "participant_id": participant_id,
                "trial_index": trial_idx,
                "accuracy": 1 if is_correct else 0,
                "item_feedback": item_feedback,
                "transfer_type": transfer_type,
                "phase": phase,
            })

    return rows


# ============================================================================
# Accuracy Plot Styling Helpers
# ============================================================================


def _apply_accuracy_style(
    ax,
    max_trial: int,
    add_ylabel: bool = False,
    y_max: float = 0.65,
    y_ticks: Sequence[float] | None = None,
) -> None:
    """Apply consistent styling to accuracy plots (mimics Section 2 aesthetic)."""
    right_limit = max(5, max_trial + 1)
    tick_limit = max(25, right_limit)

    ax.set_xlabel("Trial", fontsize=18, fontweight="bold")
    if add_ylabel:
        ax.set_ylabel("Accuracy", fontsize=18, fontweight="bold")

    ax.set_xlim(-0.5, right_limit)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(range(0, tick_limit + 1, 5))
    if y_ticks is None:
        y_ticks = [0.2, 0.4, 0.6]
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", which="major", labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_accuracy_legend(ax, title: str, loc: str = "upper right") -> None:
    legend = ax.legend(title=title, title_fontsize=16, fontsize=14,
                       frameon=False, loc=loc)
    if legend:
        plt.setp(legend.get_title(), fontweight="bold")
        for text in legend.get_texts():
            text.set_fontweight("bold")


# ============================================================================
# Plot: Learning Phase Accuracy Comparison
# ============================================================================

def plot_learning_accuracy_comparison(
    item_rows: Sequence[dict],
    no_item_rows: Sequence[dict],
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot learning accuracy with Section 2-style palette (red vs cyan)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    datasets = [
        (item_rows, "Item", ACCURACY_HIGH_COLOR),
        (no_item_rows, "No Item", ACCURACY_LOW_COLOR),
    ]

    max_trial_idx = 0

    for rows, label, color in datasets:
        if not rows:
            continue

        # Group by trial
        trial_data: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            trial_data[int(row["trial_index"])].append(int(row["accuracy"]))

        trial_indices = sorted(trial_data.keys())
        means, sems = _compute_trial_statistics(trial_data)

        max_trial_idx = max(max_trial_idx, trial_indices[-1]) if trial_indices else max_trial_idx

        ax.errorbar(
            trial_indices, means, yerr=sems,
            fmt="-o", color=color, label=label,
            linewidth=1.5, elinewidth=1.5, markersize=6, capsize=0,
        )

    ax.set_title("Learning Phase", fontweight="bold", fontsize=18)
    _apply_accuracy_style(ax, max_trial_idx, add_ylabel=True)
    _format_accuracy_legend(ax, title="Condition", loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig, ax


# ============================================================================
# Plot: Transfer Phase Accuracy Comparison
# ============================================================================

def plot_transfer_accuracy_comparison(
    item_high_rows: Sequence[dict],
    no_item_high_rows: Sequence[dict],
    item_low_rows: Sequence[dict],
    no_item_low_rows: Sequence[dict],
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot transfer accuracy: 2 subplots (high/low) using Section 2-style palette."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    subplot_data = [
        (item_high_rows, no_item_high_rows, "Manipulation High", axes[0]),
        (item_low_rows, no_item_low_rows, "Manipulation Low", axes[1]),
    ]

    for item_rows, no_item_rows, title, ax in subplot_data:
        datasets = [
            (item_rows, "Item", ACCURACY_HIGH_COLOR),
            (no_item_rows, "No Item", ACCURACY_LOW_COLOR),
        ]

        max_trial_idx = 0

        for rows, label, color in datasets:
            if not rows:
                continue

            trial_data: dict[int, list[int]] = defaultdict(list)
            for row in rows:
                trial_data[int(row["trial_index"])].append(int(row["accuracy"]))

            trial_indices = sorted(trial_data.keys())
            means, sems = _compute_trial_statistics(trial_data)

            max_trial_idx = max(max_trial_idx, trial_indices[-1]) if trial_indices else max_trial_idx

            ax.errorbar(
                trial_indices, means, yerr=sems,
                fmt="-o", color=color, label=label,
                linewidth=1.5, elinewidth=1.5, markersize=6, capsize=0,
            )

        ax.set_title(title, fontweight="bold", fontsize=18)
        _apply_accuracy_style(
            ax,
            max_trial_idx,
            add_ylabel=(title == "Manipulation High"),
            y_max=0.35,
            y_ticks=[0.1, 0.2, 0.3],
        )
        _format_accuracy_legend(ax, title="Condition", loc="upper right")

    fig.suptitle("Transfer Phase", fontweight="bold", fontsize=18, y=1.03)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig, axes


# ============================================================================
# Regression: Learning Accuracy Model
# ============================================================================

def fit_learning_accuracy_model(rows: Sequence[dict]):
    """Run mixed-effects regression on learning phase accuracy.

    Formula: accuracy ~ F + T + F:T + (1|participant)

    Where:
    - F = item_feedback (0=no_item, 1=original)
    - T = trial_centered
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    learning_rows = [r for r in rows if r.get("phase") == "learning"]
    if not learning_rows:
        raise ValueError("No learning phase data found")

    mean_trial = mean(r["trial_index"] for r in learning_rows)
    for r in learning_rows:
        r["trial_centered"] = r["trial_index"] - mean_trial

    reg = MixedRegressor(learning_rows, participant_col="participant_id")
    formula_terms = ["item_feedback", "trial_centered", "item_feedback:trial_centered"]

    try:
        return reg.fit("accuracy", formula_terms)
    except Exception as exc:
        LOG.warning("Mixed model failed (%s); retrying with powell", exc)

    try:
        return reg.fit("accuracy", formula_terms, method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed (%s); falling back to OLS", exc2)

    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(learning_rows)
    formula = "accuracy ~ item_feedback + trial_centered + item_feedback:trial_centered"
    model = smf.ols(formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


# ============================================================================
# Regression: Transfer Accuracy Model
# ============================================================================

def fit_transfer_accuracy_model(rows: Sequence[dict]):
    """Run mixed-effects regression on transfer phase accuracy.

    Formula: accuracy ~ F + T + L + F:T + F:L + F:T:L + (1|participant)

    Where:
    - F = item_feedback (0=no_item, 1=original)
    - T = trial_centered
    - L = transfer_type (0=low, 1=high)
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    transfer_rows = [r for r in rows if r.get("phase") == "transfer"]
    if not transfer_rows:
        raise ValueError("No transfer phase data found")

    mean_trial = mean(r["trial_index"] for r in transfer_rows)
    for r in transfer_rows:
        r["trial_centered"] = r["trial_index"] - mean_trial
        r["transfer_type_num"] = 1 if r["transfer_type"] == "high" else 0

    reg = MixedRegressor(transfer_rows, participant_col="participant_id")
    formula_terms = [
        "item_feedback", "trial_centered", "transfer_type_num",
        "item_feedback:trial_centered", "item_feedback:transfer_type_num",
        "item_feedback:trial_centered:transfer_type_num",
    ]

    try:
        return reg.fit("accuracy", formula_terms)
    except Exception as exc:
        LOG.warning("Mixed model failed (%s); retrying with powell", exc)

    try:
        return reg.fit("accuracy", formula_terms, method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed (%s); falling back to OLS", exc2)

    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(transfer_rows)
    formula = "accuracy ~ item_feedback + trial_centered + transfer_type_num + item_feedback:trial_centered + item_feedback:transfer_type_num + item_feedback:trial_centered:transfer_type_num"
    model = smf.ols(formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


# ============================================================================
# Regression Model 2: Transfer Model
# ============================================================================

def fit_transfer_model(rows: Sequence[dict]):
    """Run mixed-effects regression on transfer phase data.

    Formula: count ~ V + T + F + L + V:T + V:F + V:T:L + V:T:F:L + (1|participant)

    Where:
    - V = validity (0=invalid, 1=valid)
    - T = trial_centered
    - F = item_feedback (0=no_item, 1=original)
    - L = transfer_type (0=low, 1=high)
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    # Filter to transfer phase only
    transfer_rows = [r for r in rows if r.get("phase") == "transfer"]
    if not transfer_rows:
        raise ValueError("No transfer phase data found")

    # Center trial index and encode transfer_type
    mean_trial = mean(r["trial_index"] for r in transfer_rows)
    for r in transfer_rows:
        r["trial_centered"] = r["trial_index"] - mean_trial
        r["transfer_type_num"] = 1 if r["transfer_type"] == "high" else 0

    reg = MixedRegressor(transfer_rows, participant_col="participant_id")
    formula_terms = [
        "validity", "trial_centered", "item_feedback", "transfer_type_num",
        "validity:trial_centered", "validity:item_feedback",
        "validity:trial_centered:transfer_type_num",
        "validity:trial_centered:item_feedback:transfer_type_num"
    ]

    # Try mixed model with primary optimizer
    try:
        return reg.fit("count", formula_terms)
    except Exception as exc:
        LOG.warning("Mixed model failed (%s); retrying with powell", exc)

    # Try alternative optimizer
    try:
        return reg.fit("count", formula_terms, method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed (%s); falling back to OLS", exc2)

    # Fallback to clustered OLS
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(transfer_rows)
    formula = "count ~ validity + trial_centered + item_feedback + transfer_type_num + validity:trial_centered + validity:item_feedback + validity:trial_centered:transfer_type_num + validity:trial_centered:item_feedback:transfer_type_num"
    model = smf.ols(formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    """CLI entrypoint for item stimulus experiment analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Analyze item stimulus effect on 2-key sequence chunking."
    )
    parser.add_argument("--original_high", required=True, help="Path to original transfer_high experiment.json")
    parser.add_argument("--original_low", required=True, help="Path to original transfer_low experiment.json")
    parser.add_argument("--no_item_high", required=True, help="Path to no_item transfer_high experiment.json")
    parser.add_argument("--no_item_low", required=True, help="Path to no_item transfer_low experiment.json")
    parser.add_argument("--output_dir", default="analysis/plots", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    # Load all four conditions
    participants = load_four_conditions(
        Path(args.original_high),
        Path(args.original_low),
        Path(args.no_item_high),
        Path(args.no_item_low),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse sequences and correctness for all conditions
    all_rows: list[dict] = []
    accuracy_rows: list[dict] = []

    for condition, participants_list in participants.items():
        sequences = parse_action_sequence(participants_list)
        correctness = parse_trial_correctness(participants_list)

        # Determine parameters
        item_feedback = 1 if "original" in condition else 0
        transfer_type = "high" if "high" in condition else "low"

        # Process learning phase (2-key sequences)
        learning_rows = count_valid_invalid_2seq_blocks(
            sequences["learning_trials"],
            correctness["learning_trials"],
            item_feedback=item_feedback,
            transfer_type=transfer_type,
            phase="learning",
            num_blocks=2,
        )
        all_rows.extend(learning_rows)
        LOG.info("%s learning: %s rows", condition, len(learning_rows))

        # Process transfer phase (2-key sequences)
        transfer_rows = count_valid_invalid_2seq_blocks(
            sequences["transfer_trials"],
            correctness["transfer_trials"],
            item_feedback=item_feedback,
            transfer_type=transfer_type,
            phase="transfer",
            num_blocks=2,
        )
        all_rows.extend(transfer_rows)
        LOG.info("%s transfer: %s rows", condition, len(transfer_rows))

        # Collect accuracy rows for new experiments (first 40 trials = 4 blocks)
        acc_learning = collect_accuracy_rows(
            correctness["learning_trials"],
            item_feedback=item_feedback,
            transfer_type=transfer_type,
            phase="learning",
            num_blocks=4,
        )
        accuracy_rows.extend(acc_learning)

        acc_transfer = collect_accuracy_rows(
            correctness["transfer_trials"],
            item_feedback=item_feedback,
            transfer_type=transfer_type,
            phase="transfer",
            num_blocks=4,
        )
        accuracy_rows.extend(acc_transfer)

    # Separate data for plotting
    # Learning: use transfer_low for both (per specification)
    original_learning = [r for r in all_rows if r["phase"] == "learning" and r["item_feedback"] == 1 and r["transfer_type"] == "low"]
    no_item_learning = [r for r in all_rows if r["phase"] == "learning" and r["item_feedback"] == 0 and r["transfer_type"] == "low"]

    # Transfer: all 4 combinations
    original_high_transfer = [r for r in all_rows if r["phase"] == "transfer" and r["item_feedback"] == 1 and r["transfer_type"] == "high"]
    original_low_transfer = [r for r in all_rows if r["phase"] == "transfer" and r["item_feedback"] == 1 and r["transfer_type"] == "low"]
    no_item_high_transfer = [r for r in all_rows if r["phase"] == "transfer" and r["item_feedback"] == 0 and r["transfer_type"] == "high"]
    no_item_low_transfer = [r for r in all_rows if r["phase"] == "transfer" and r["item_feedback"] == 0 and r["transfer_type"] == "low"]

    # Generate Plot 1: Learning Phase
    LOG.info("=" * 60)
    LOG.info("PLOT 1: Learning Phase Comparison")
    LOG.info("=" * 60)
    plot_path = output_dir / "item_stimulus_learning.png"
    plot_learning_phase_comparison(
        original_learning, no_item_learning,
        save_path=plot_path, show=args.show
    )
    LOG.info("Saved learning plot to %s", plot_path)

    # Generate Plot 2: Transfer Phase
    LOG.info("=" * 60)
    LOG.info("PLOT 2: Transfer Phase All Conditions")
    LOG.info("=" * 60)
    plot_path = output_dir / "item_stimulus_transfer.png"
    plot_transfer_phase_all_conditions(
        original_high_transfer, original_low_transfer,
        no_item_high_transfer, no_item_low_transfer,
        save_path=plot_path, show=args.show
    )
    LOG.info("Saved transfer plot to %s", plot_path)

    # Fit Regression Model 1: Learning Only
    LOG.info("=" * 60)
    LOG.info("REGRESSION 1: Learning Phase Model")
    LOG.info("=" * 60)
    try:
        result, summary = fit_learning_model(all_rows)
        LOG.info("Learning model complete")
        LOG.info("\n%s", summary.to_string(index=False))
    except Exception as exc:
        LOG.warning("Learning regression failed: %s", exc)

    # Fit Regression Model 2: Transfer Model
    LOG.info("=" * 60)
    LOG.info("REGRESSION 2: Transfer Phase Model")
    LOG.info("=" * 60)
    try:
        result, summary = fit_transfer_model(all_rows)
        LOG.info("Transfer model complete")
        LOG.info("\n%s", summary.to_string(index=False))
    except Exception as exc:
        LOG.warning("Transfer regression failed: %s", exc)

    # =========================================================================
    # NEW EXPERIMENTS: Item Stimulus Effect on Accuracy
    # =========================================================================

    # Separate accuracy data for plotting
    # Learning: use transfer_high (per specification)
    item_learning_acc = [r for r in accuracy_rows if r["phase"] == "learning" and r["item_feedback"] == 1 and r["transfer_type"] == "high"]
    no_item_learning_acc = [r for r in accuracy_rows if r["phase"] == "learning" and r["item_feedback"] == 0 and r["transfer_type"] == "high"]

    # Transfer: all 4 combinations
    item_high_transfer_acc = [r for r in accuracy_rows if r["phase"] == "transfer" and r["item_feedback"] == 1 and r["transfer_type"] == "high"]
    item_low_transfer_acc = [r for r in accuracy_rows if r["phase"] == "transfer" and r["item_feedback"] == 1 and r["transfer_type"] == "low"]
    no_item_high_transfer_acc = [r for r in accuracy_rows if r["phase"] == "transfer" and r["item_feedback"] == 0 and r["transfer_type"] == "high"]
    no_item_low_transfer_acc = [r for r in accuracy_rows if r["phase"] == "transfer" and r["item_feedback"] == 0 and r["transfer_type"] == "low"]

    # Plot 3: Learning Accuracy Comparison
    LOG.info("=" * 60)
    LOG.info("PLOT 3: Learning Phase Accuracy (Item vs No Item)")
    LOG.info("=" * 60)
    plot_path = output_dir / "item_stimulus_learning_accuracy.png"
    plot_learning_accuracy_comparison(
        item_learning_acc, no_item_learning_acc,
        save_path=plot_path, show=args.show
    )
    LOG.info("Saved learning accuracy plot to %s", plot_path)

    # Plot 4: Transfer Accuracy Comparison
    LOG.info("=" * 60)
    LOG.info("PLOT 4: Transfer Phase Accuracy (Item vs No Item)")
    LOG.info("=" * 60)
    plot_path = output_dir / "item_stimulus_transfer_accuracy.png"
    plot_transfer_accuracy_comparison(
        item_high_transfer_acc, no_item_high_transfer_acc,
        item_low_transfer_acc, no_item_low_transfer_acc,
        save_path=plot_path, show=args.show
    )
    LOG.info("Saved transfer accuracy plot to %s", plot_path)

    # Regression 3: Learning Accuracy Model
    LOG.info("=" * 60)
    LOG.info("REGRESSION 3: Learning Accuracy Model")
    LOG.info("=" * 60)
    try:
        result, summary = fit_learning_accuracy_model(accuracy_rows)
        LOG.info("Learning accuracy model complete")
        LOG.info("\n%s", summary.to_string(index=False))
    except Exception as exc:
        LOG.warning("Learning accuracy regression failed: %s", exc)

    # Regression 4: Transfer Accuracy Model
    LOG.info("=" * 60)
    LOG.info("REGRESSION 4: Transfer Accuracy Model")
    LOG.info("=" * 60)
    try:
        result, summary = fit_transfer_accuracy_model(accuracy_rows)
        LOG.info("Transfer accuracy model complete")
        LOG.info("\n%s", summary.to_string(index=False))
    except Exception as exc:
        LOG.warning("Transfer accuracy regression failed: %s", exc)

    LOG.info("=" * 60)
    LOG.info("Analysis complete!")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()

"""
Sample usage:
    # Run item vs no_item comparison across all 4 conditions
    PYTHONPATH=. python analysis/new_experiments/item_stimulus.py \
        --original_high results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json \
        --original_low results/gpt-oss-120b/transfer_low/item_20251122_040528_experiment.json \
        --no_item_high results/gpt-oss-120b/transfer_high/no_item_20251122_075552_experiment.json \
        --no_item_low results/gpt-oss-120b/transfer_low/no_item_20251122_040337_experiment.json \
        --output_dir results/gpt-oss-120b/experiments/item_stimulus

    # With interactive display
    PYTHONPATH=. python analysis/new_experiments/item_stimulus.py \
        --original_high results/model/transfer_high/item.json \
        --original_low results/model/transfer_low/item.json \
        --no_item_high results/model/transfer_high/no_item.json \
        --no_item_low results/model/transfer_low/no_item.json \
        --output_dir results/model/experiments/item_stimulus \
        --show
"""
