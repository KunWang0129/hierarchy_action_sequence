"""
Implementation of analysis for the following experiment:
Files to load: 4 paths:
- original transfer high
- original transfer low
- generalize transfer high
- generalize transfer low

# Experiment 1: Do generalizing improve transfer performance compared to original transfer? Is their an interaction with transfer type (high vs low)?
Question: Do model perform better in the transfer phase when generalize with new stimuli? Does generalize improve more in high transfer than low transfer?
Predictors:
Generalize (G) (True/False)
Trial numbers (T)
TransferType (L)
Plot:
Plot content: accuracy over trials in transfer phase
Red line: original, blue line: generalize
First 40 trials in transfer
One plot with 2 subplots: transfer high and transfer low
Model: mixed-effect regressor predicting accuracy
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

from analysis.functions import (  # noqa: E402
    MixedRegressor,
    PLOT_STYLE,
    _create_summary_dataframe,
    _flatten_blocks,
    parse_trial_correctness,
    setup_plot_style,
)

setup_plot_style()
LOG = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_conditions(
    original_high_path: Path,
    original_low_path: Path,
    generalize_high_path: Path,
    generalize_low_path: Path,
) -> dict[str, list[dict]]:
    """Load participants from all four experimental conditions."""
    paths = {
        "original_high": original_high_path,
        "original_low": original_low_path,
        "generalize_high": generalize_high_path,
        "generalize_low": generalize_low_path,
    }

    data: dict[str, list[dict]] = {}
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data[label] = payload.get("participants", [])
        LOG.info("Loaded %s participants from %s", len(data[label]), label)

    return data


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_trial_statistics(data_dict: dict[int, list[int]]) -> tuple[list[float], list[float]]:
    """Compute means and SEMs for per-trial data."""
    means, sems = [], []
    for t_idx in sorted(data_dict.keys()):
        vals = data_dict.get(t_idx, [])
        means.append(mean(vals) if vals else 0.0)
        sems.append(0.0 if len(vals) <= 1 else pstdev(vals) / sqrt(len(vals)))
    return means, sems


def collect_transfer_accuracy(
    correctness: dict[int, list[list[bool]]],
    generalize: int,
    transfer_type: str,
    num_blocks: int = 4,
) -> list[dict]:
    """Collect per-trial accuracy rows from the transfer phase."""
    rows: list[dict] = []

    for participant_id, blocks in correctness.items():
        selected_blocks = blocks[:num_blocks]
        trials = _flatten_blocks(selected_blocks)
        for trial_idx, is_correct in enumerate(trials):
            rows.append({
                "participant_id": participant_id,
                "trial_index": trial_idx,
                "accuracy": 1 if is_correct else 0,
                "generalize": generalize,
                "transfer_type": transfer_type,
                "phase": "transfer",
            })

    return rows


# ============================================================================
# Plotting
# ============================================================================

def plot_transfer_accuracy(
    original_high_rows: Sequence[dict],
    generalize_high_rows: Sequence[dict],
    original_low_rows: Sequence[dict],
    generalize_low_rows: Sequence[dict],
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot transfer accuracy for high and low conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    datasets = [
        (original_high_rows, generalize_high_rows, "Manipulation High", axes[0]),
        (original_low_rows, generalize_low_rows, "Manipulation Low", axes[1]),
    ]

    color_original = "#E15759"    # red
    color_generalize = "#4EBFD8"  # cyan

    for original_rows, generalize_rows, title, ax in datasets:
        trial_data_original: dict[int, list[int]] = defaultdict(list)
        trial_data_generalize: dict[int, list[int]] = defaultdict(list)

        for row in original_rows:
            trial_data_original[int(row["trial_index"])].append(int(row["accuracy"]))
        for row in generalize_rows:
            trial_data_generalize[int(row["trial_index"])].append(int(row["accuracy"]))

        for trial_data, label, color in [
            (trial_data_original, "Original", color_original),
            (trial_data_generalize, "Generalize", color_generalize),
        ]:
            if not trial_data:
                continue
            trial_indices = sorted(trial_data.keys())
            means, sems = _compute_trial_statistics(trial_data)
            ax.errorbar(
                trial_indices,
                means,
                yerr=sems,
                fmt="-o",
                color=color,
                label=label,
                linewidth=1.5,
                markersize=6,
                capsize=0,
                elinewidth=1.5,
            )

        ax.set_xlabel("Trial", fontsize=20, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy", fontsize=20, fontweight="bold")
        ax.set_ylim(0.0, 0.35)
        ax.set_xlim(-0.5, 40)
        ax.set_title(title, fontweight="bold", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=16)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(range(0, 41, 10))
        ax.set_yticks([0.1, 0.2, 0.3])
        ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = axes[1].legend(
        handles,
        labels,
        title="Condition",
        title_fontsize=18,
        fontsize=16,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    plt.setp(legend.get_title(), fontweight="bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # fig.suptitle(
    #     "Transfer Accuracy: Original vs Generalize",
    #     fontweight="bold",
    #     y=1.02,
    #     fontsize=20,
    # )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig, axes


# ============================================================================
# Regression
# ============================================================================

def fit_transfer_accuracy_model(rows: Sequence[dict]):
    """Run mixed-effects regression on transfer accuracy.

    Formula: accuracy ~ G + T + L + G:T + G:L + G:T:L + (1|participant)

    Where:
        G = generalize (0=original, 1=generalize)
        T = trial_centered
        L = transfer_type (0=low, 1=high)
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
        "generalize",
        "trial_centered",
        "transfer_type_num",
        "generalize:trial_centered",
        "generalize:transfer_type_num",
        "generalize:trial_centered:transfer_type_num",
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
    formula = (
        "accuracy ~ generalize + trial_centered + transfer_type_num + "
        "generalize:trial_centered + generalize:transfer_type_num + "
        "generalize:trial_centered:transfer_type_num"
    )
    model = smf.ols(formula, data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    """CLI entrypoint for generalize vs original transfer analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Analyze transfer accuracy for generalize vs original stimuli.")
    parser.add_argument("--original_high", required=True, help="Path to original transfer_high experiment.json")
    parser.add_argument("--original_low", required=True, help="Path to original transfer_low experiment.json")
    parser.add_argument("--generalize_high", required=True, help="Path to generalize transfer_high experiment.json")
    parser.add_argument("--generalize_low", required=True, help="Path to generalize transfer_low experiment.json")
    parser.add_argument("--output_dir", default="analysis/plots", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    participants = load_conditions(
        Path(args.original_high),
        Path(args.original_low),
        Path(args.generalize_high),
        Path(args.generalize_low),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for condition, participants_list in participants.items():
        correctness = parse_trial_correctness(participants_list)["transfer_trials"]
        generalize_flag = 1 if "generalize" in condition else 0
        transfer_type = "high" if "high" in condition else "low"

        transfer_rows = collect_transfer_accuracy(
            correctness,
            generalize=generalize_flag,
            transfer_type=transfer_type,
            num_blocks=4,
        )
        all_rows.extend(transfer_rows)
        LOG.info("%s transfer rows: %s", condition, len(transfer_rows))

    original_high_rows = [r for r in all_rows if r["transfer_type"] == "high" and r["generalize"] == 0]
    generalize_high_rows = [r for r in all_rows if r["transfer_type"] == "high" and r["generalize"] == 1]
    original_low_rows = [r for r in all_rows if r["transfer_type"] == "low" and r["generalize"] == 0]
    generalize_low_rows = [r for r in all_rows if r["transfer_type"] == "low" and r["generalize"] == 1]

    LOG.info("=" * 60)
    LOG.info("PLOT: Transfer Accuracy (Original vs Generalize)")
    LOG.info("=" * 60)
    plot_path = output_dir / "generalize_transfer_accuracy.png"
    plot_transfer_accuracy(
        original_high_rows,
        generalize_high_rows,
        original_low_rows,
        generalize_low_rows,
        save_path=plot_path,
        show=args.show,
    )
    LOG.info("Saved transfer plot to %s", plot_path)

    LOG.info("=" * 60)
    LOG.info("REGRESSION: Transfer Accuracy Model")
    LOG.info("=" * 60)
    try:
        result, summary = fit_transfer_accuracy_model(all_rows)
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
    PYTHONPATH=. python analysis/new_experiments/generalize_transfer.py \
        --original_high results/gpt-oss-120b/transfer_high/original.json \
        --original_low results/gpt-oss-120b/transfer_low/original.json \
        --generalize_high results/gpt-oss-120b/transfer_high/generalize.json \
        --generalize_low results/gpt-oss-120b/transfer_low/generalize.json \
        --output_dir results/gpt-oss-120b/experiments/generalize_transfer
"""
