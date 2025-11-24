"""Learning-performance helpers for star making."""

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


def rep_valid_invalid_2_seq(
    learning_sequences: dict[int, list[list[list[int]]]],
    learning_trial_correctness: dict[int, list[list[bool]]],
    t: int = 10,
) -> list[dict]:
    """Count valid/invalid 2-key options over the first `t` learning trials.

    Only incorrect trials contribute counts; valid/invalid are assessed on the
    two action pairs ([0,1] and [2,3]) using learning rules.
    """
    rules = StarMakingRules()
    valid_pairs = set(rules.learning_rules["low"].keys())

    rows: list[dict] = []
    for participant_id, blocks in learning_sequences.items():
        seqs = _flatten_blocks(blocks)
        correctness = _flatten_blocks(learning_trial_correctness.get(participant_id, []))

        if len(seqs) != len(correctness):
            LOG.warning(
                "Sequence/correctness length mismatch for participant %s (%s vs %s)",
                participant_id,
                len(seqs),
                len(correctness),
            )
        n = min(len(seqs), len(correctness), t)
        for trial_idx in range(n):
            if correctness[trial_idx]:
                continue  # skip correct trials
            seq = seqs[trial_idx]
            if len(seq) < 4:
                LOG.debug("Skipping short sequence for participant %s trial %s: %s", participant_id, trial_idx, seq)
                continue
            pairs = [(seq[0], seq[1]), (seq[2], seq[3])]
            valid_count = sum(1 for pair in pairs if pair in valid_pairs)
            invalid_count = sum(1 for pair in pairs if pair in INVALID_2_SEQ)
            rows.append(
                {
                    "participant_id": participant_id,
                    "trial_index": trial_idx,
                    "valid_count": valid_count,
                    "invalid_count": invalid_count,
                }
            )
    return rows


def fit_valid_invalid_regression(rows: Sequence[dict]):
    """Run mixed-effects regression on valid vs invalid counts per trial.

    Falls back to cluster-robust OLS if the mixed model is singular.
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    long_rows: list[dict] = []
    for row in rows:
        pid = row["participant_id"]
        t_idx = int(row["trial_index"])
        long_rows.append({"participant_id": pid, "trial_index": t_idx, "validity": 1, "count": row["valid_count"]})
        long_rows.append({"participant_id": pid, "trial_index": t_idx, "validity": 0, "count": row["invalid_count"]})

    if not long_rows:
        raise ValueError("No counts available to fit regression")

    mean_trial = mean(r["trial_index"] for r in long_rows)
    for r in long_rows:
        r["trial_centered"] = r["trial_index"] - mean_trial

    reg = MixedRegressor(long_rows, participant_col="participant_id")
    formula_terms = ["validity", "trial_centered", "validity:trial_centered"]

    # Try mixed model with primary optimizer
    try:
        return reg.fit("count", formula_terms)
    except Exception as exc:
        LOG.warning("Mixed model failed (%s); retrying with alternative optimizer", exc)

    # Try alternative optimizer
    try:
        return reg.fit("count", formula_terms, method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed (%s); falling back to clustered OLS", exc2)

    # Fallback to clustered OLS
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(long_rows)
    model = smf.ols("count ~ validity + trial_centered + validity:trial_centered", data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


def _fit_count_regression(rows: Sequence[dict], count_col: str, min_threshold: int = 1):
    """Generic mixed-effects regression for count subsets.

    Args:
        rows: Trial data from rep_valid_invalid_2_seq
        count_col: Column to regress ("valid_count" or "invalid_count")
        min_threshold: Minimum count value to include

    Returns:
        Tuple of (statsmodels result, summary dataframe)
    """
    if not rows:
        raise ValueError("No data supplied for regression")

    # Filter rows by count threshold
    filtered = [r for r in rows if r.get(count_col, 0) >= min_threshold]
    if not filtered:
        raise ValueError(f"No rows with {count_col} >= {min_threshold}")

    # Center trial index
    mean_trial = mean(r["trial_index"] for r in filtered)
    for r in filtered:
        r["trial_centered"] = r["trial_index"] - mean_trial

    reg = MixedRegressor(filtered, participant_col="participant_id")

    # Try mixed model with primary optimizer
    try:
        return reg.fit(count_col, ["trial_centered"])
    except Exception as exc:
        LOG.warning("Mixed model failed for %s (%s); retrying with alternative optimizer", count_col, exc)

    # Try alternative optimizer
    try:
        return reg.fit(count_col, ["trial_centered"], method="powell")
    except Exception as exc2:
        LOG.warning("Mixed model still failed for %s (%s); falling back to clustered OLS", count_col, exc2)

    # Fallback to clustered OLS
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(filtered)
    model = smf.ols(f"{count_col} ~ trial_centered", data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["participant_id"]})
    return result, _create_summary_dataframe(result)


def fit_valid_only_regression(rows: Sequence[dict]):
    """Run mixed-effects regression on valid counts (valid_count >= 1).

    Formula: valid_count ~ trial_centered + (1|participant_id)
    """
    return _fit_count_regression(rows, "valid_count", min_threshold=1)


def fit_invalid_only_regression(rows: Sequence[dict]):
    """Run mixed-effects regression on invalid counts (invalid_count >= 1).

    Formula: invalid_count ~ trial_centered + (1|participant_id)
    """
    return _fit_count_regression(rows, "invalid_count", min_threshold=1)


def _compute_trial_statistics(
    data_dict: dict[int, list[int]],
    trial_order: Sequence[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute means and SEMs for trial data following the requested order."""

    if trial_order is None:
        trial_order = sorted(data_dict.keys())

    means, sems = [], []
    for t_idx in trial_order:
        vals = data_dict.get(t_idx, [])
        means.append(mean(vals) if vals else 0.0)
        sems.append(0.0 if len(vals) <= 1 else pstdev(vals) / sqrt(len(vals)))
    return means, sems


def plot_valid_invalid_counts(
    rows: Sequence[dict],
    ax=None,
    show: bool = False,
    save_path: str | Path | None = None,
):
    """Plot mean valid/invalid counts per trial with SEM error bars."""
    if not rows:
        LOG.warning("No data to plot")
        return None, None

    counts = {"valid": defaultdict(list), "invalid": defaultdict(list)}
    for row in rows:
        t_idx = int(row["trial_index"])
        counts["valid"][t_idx].append(int(row["valid_count"]))
        counts["invalid"][t_idx].append(int(row["invalid_count"]))

    trial_indices = sorted(set(counts["valid"]) | set(counts["invalid"]))
    stats = {label: _compute_trial_statistics(bucket, trial_indices) for label, bucket in counts.items()}

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        tight_layout = True
    else:
        fig = ax.figure
        tight_layout = False

    for label in ["valid", "invalid"]:
        means, sems = stats[label]
        if not trial_indices:
            continue
        lower = [max(m - s, 0) for m, s in zip(means, sems)]
        upper = [m + s for m, s in zip(means, sems)]
        ax.fill_between(trial_indices, lower, upper, color=PLOT_COLORS[label],
                        alpha=PLOT_STYLE["fill_alpha"], linewidth=0)
        ax.errorbar(trial_indices, means, yerr=sems,
                    fmt=f"{PLOT_MARKERS[label]}-", color=PLOT_COLORS[label],
                    linewidth=PLOT_STYLE["linewidth"], markersize=PLOT_STYLE["markersize"],
                    capsize=PLOT_STYLE["capsize"], label=label.capitalize())

    ax.set_xlabel("Trial index")
    ax.set_ylabel("Count")
    ax.set_title("Valid vs invalid 2-key sequences (learning, incorrect trials)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    if tight_layout:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig, ax


def main():
    """CLI entrypoint to run valid/invalid analysis and plot."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    default_path = Path("results") / "gpt-oss-120b" / "transfer_low" / "20251119_162228_experiment.json"
    parser = argparse.ArgumentParser(description="Analyze valid/invalid 2-key usage in learning trials.")
    parser.add_argument("input_path", nargs="?", default=str(default_path), help="Path to *_experiment.json")
    parser.add_argument("--output", "-o", help="Optional path to save the plot (e.g., plot.png)")
    parser.add_argument("--trials", "-t", type=int, default=10, help="Number of initial learning trials to include")
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    participants = data.get("participants", [])
    LOG.info("Loaded %s participants from %s", len(participants), input_path)

    sequences = parse_action_sequence(participants)["learning_trials"]
    correctness = parse_trial_correctness(participants)["learning_trials"]
    rows = rep_valid_invalid_2_seq(sequences, correctness, t=args.trials)
    LOG.info("Computed %s trial rows (incorrect only)", len(rows))

    try:
        result, summary = fit_valid_invalid_regression(rows)
        LOG.info("Mixed-effects regression complete")
        LOG.info("\n%s", summary.to_string(index=False))
    except Exception as exc:  # pragma: no cover - runtime safeguard
        LOG.warning("Regression failed: %s", exc)

    # Valid-only regression
    try:
        valid_result, valid_summary = fit_valid_only_regression(rows)
        LOG.info("Valid-only regression complete")
        LOG.info("\n%s", valid_summary.to_string(index=False))
    except ValueError as exc:
        LOG.warning("Valid-only regression skipped: %s", exc)
    except Exception as exc:
        LOG.warning("Valid-only regression failed: %s", exc)

    # Invalid-only regression
    try:
        invalid_result, invalid_summary = fit_invalid_only_regression(rows)
        LOG.info("Invalid-only regression complete")
        LOG.info("\n%s", invalid_summary.to_string(index=False))
    except ValueError as exc:
        LOG.warning("Invalid-only regression skipped: %s", exc)
    except Exception as exc:
        LOG.warning("Invalid-only regression failed: %s", exc)

    plot_valid_invalid_counts(rows, show=args.show, save_path=args.output)


if __name__ == "__main__":
    main()


"""
Sample usage:
    # Run for all 4 conditions
    python analysis/original_experiments/learning_phase.py \
        results/gpt-oss-120b/transfer_low/no_item_20251122_040337_experiment.json \
        -o results/gpt-oss-120b/experiments/learning_phase/transfer_low/no_item/learning_valid_invalid.png --trials 20

    python analysis/original_experiments/learning_phase.py \
        results/gpt-oss-120b/transfer_low/item_20251122_040528_experiment.json \
        -o results/gpt-oss-120b/experiments/learning_phase/transfer_low/item/learning_valid_invalid.png --trials 20

    python analysis/original_experiments/learning_phase.py \
        results/gpt-oss-120b/transfer_high/no_item_20251122_075552_experiment.json \
        -o results/gpt-oss-120b/experiments/learning_phase/transfer_high/no_item/learning_valid_invalid.png --trials 20

    python analysis/original_experiments/learning_phase.py \
        results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json \
        -o results/gpt-oss-120b/experiments/learning_phase/transfer_high/item/learning_valid_invalid.png --trials 20

    # With interactive display
    python analysis/original_experiments/learning_phase.py path/to/experiment.json -o output.png --trials 20 --show
"""
