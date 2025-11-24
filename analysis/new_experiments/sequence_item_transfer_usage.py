"""Transfer-phase analysis of 4-key reuse and 2-key chunk preference.

Usage example:

python analysis/new_experiments/sequence_item_transfer_usage.py \
    --transfer-high results/gpt-oss-120b/transfer_high/item_20251120_220149_experiment.json \
    --transfer-low results/gpt-oss-120b/transfer_low/item_20251120_220149_experiment.json \
    --output-dir analysis/plots/sequence_item_transfer_usage
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
from typing import Callable, Sequence

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.functions import (  # noqa: E402
    MixedRegressor,
    PLOT_COLORS,
    PLOT_MARKERS,
    PLOT_STYLE,
    _create_summary_dataframe,
    _flatten_blocks,
    parse_action_sequence,
    parse_goal_stars,
    parse_trial_correctness,
    setup_plot_style,
)
from inference.star_making.assets_utils import StarMakingRules  # noqa: E402

LOG = logging.getLogger(__name__)
INVALID_2_SEQ = {(0, 2), (2, 1), (1, 3), (3, 1)}

setup_plot_style()


def load_participants(path: Path) -> Sequence[dict]:
    """Read experiment JSON and return raw participant entries."""
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    participants = payload.get("participants", [])
    LOG.info("Loaded %s participants from %s", len(participants), path)
    return participants


def parse_condition(participants: Sequence[dict]) -> dict[str, dict]:
    """Run shared parsers to obtain per-phase sequences, correctness, and goals."""
    sequences = parse_action_sequence(participants)
    correctness = parse_trial_correctness(participants)
    goals = parse_goal_stars(participants)
    return {
        "learning_sequences": sequences["learning_trials"],
        "transfer_sequences": sequences["transfer_trials"],
        "learning_correctness": correctness["learning_trials"],
        "transfer_correctness": correctness["transfer_trials"],
        "learning_goals": goals["learning_trials"],
        "transfer_goals": goals["transfer_trials"],
    }


def build_learning_memory(
    learning_sequences: dict,
    learning_correctness: dict,
    learning_goals: dict,
) -> dict:
    """Capture rewarded 4-key sequences per star for each participant."""
    memory: dict = {}

    for participant_id, seq_blocks in learning_sequences.items():
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


def label_transfer_sequences(
    transfer_sequences: dict,
    transfer_goals: dict,
    learning_memory: dict,
    condition_label: str,
    max_trials: int | None,
) -> list[dict]:
    """Label transfer-phase sequences as OLD or NEW relative to learning memory."""
    rows: list[dict] = []

    for participant_id, seq_blocks in transfer_sequences.items():
        seqs = _flatten_blocks(seq_blocks)
        goals = _flatten_blocks(transfer_goals.get(participant_id, []))
        if not seqs:
            continue

        limit = min(len(seqs), max_trials) if max_trials else len(seqs)
        memo = learning_memory.get(participant_id, {"per_star": {}, "all_sequences": set()})
        per_star = memo.get("per_star", {})
        all_sequences = memo.get("all_sequences", set())

        for idx in range(limit):
            seq = seqs[idx]
            if len(seq) < 4:
                continue

            seq_tuple = tuple(seq[:4])
            goal = goals[idx] if idx < len(goals) else None
            goal = goal or "Unknown"
            this_star_sequences = per_star.get(goal, set())
            other_sequences = all_sequences - this_star_sequences if all_sequences else set()

            if seq_tuple in this_star_sequences:
                label = "OLD_CORRECT_THIS"
            elif seq_tuple in other_sequences:
                label = "OLD_CORRECT_OTHER"
            else:
                label = "NEW"

            rows.append({
                "participant_id": participant_id,
                "trial_index": idx + 1,
                "condition": condition_label,
                "goal_star": goal,
                "sequence": seq_tuple,
                "label": label,
                "old_flag": 0 if label == "NEW" else 1,
            })

    LOG.info("Labeled %s transfer trials for %s", len(rows), condition_label)
    return rows


def count_two_key_usage(
    transfer_sequences: dict,
    condition_label: str,
    valid_pairs: set[tuple[int, int]],
    max_trials: int | None,
) -> list[dict]:
    """Count valid-vs-invalid 2-key pairs per trial."""
    rows: list[dict] = []

    for participant_id, seq_blocks in transfer_sequences.items():
        seqs = _flatten_blocks(seq_blocks)
        if not seqs:
            continue

        limit = min(len(seqs), max_trials) if max_trials else len(seqs)

        for idx in range(limit):
            seq = seqs[idx]
            if len(seq) < 4:
                continue

            pairs = [(seq[0], seq[1]), (seq[2], seq[3])]
            valid_count = sum(1 for pair in pairs if pair in valid_pairs)
            invalid_count = sum(1 for pair in pairs if pair in INVALID_2_SEQ)

            rows.append({
                "participant_id": participant_id,
                "trial_index": idx + 1,
                "condition": condition_label,
                "seq_type": "valid_prev",
                "seq_type_indicator": 1,
                "count": valid_count,
            })
            rows.append({
                "participant_id": participant_id,
                "trial_index": idx + 1,
                "condition": condition_label,
                "seq_type": "invalid_matched",
                "seq_type_indicator": 0,
                "count": invalid_count,
            })

    LOG.info("Computed 2-key counts for %s rows in %s", len(rows), condition_label)
    return rows


def _summarize_trial_map(trial_map: dict[int, list[float]]) -> tuple[list[int], list[float], list[float]]:
    indices = sorted(trial_map.keys())
    means, sems = [], []
    for idx in indices:
        vals = trial_map[idx]
        if not vals:
            means.append(0.0)
            sems.append(0.0)
            continue
        means.append(mean(vals))
        sems.append(0.0 if len(vals) <= 1 else pstdev(vals) / sqrt(len(vals)))
    return indices, means, sems


def compute_condition_curve(
    rows: Sequence[dict],
    condition: str,
    value_getter: Callable[[dict], float],
    max_trials: int | None,
    filter_fn: Callable[[dict], bool] | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Aggregate row-wise values into per-trial means for a condition."""
    trial_map: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("condition") != condition:
            continue
        if max_trials and row["trial_index"] > max_trials:
            continue
        if filter_fn and not filter_fn(row):
            continue
        trial_map[row["trial_index"]].append(float(value_getter(row)))
    return _summarize_trial_map(trial_map)


def plot_old_sequence_probability(
    rows: Sequence[dict],
    max_trials: int | None,
    save_path: Path,
    show: bool,
) -> None:
    """Plot probability of executing an OLD sequence across transfer trials."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, color_key, label in [("transfer_high", "high", "Transfer High"), ("transfer_low", "low", "Transfer Low")]:
        trials, means, sems = compute_condition_curve(rows, condition, lambda r: float(r["old_flag"]), max_trials)
        if not trials:
            ax.plot([], [], label=f"{label} (no data)", color=PLOT_COLORS[color_key])
            continue
        ax.errorbar(trials, means, yerr=sems,
                    fmt=f"{PLOT_MARKERS[color_key]}-", color=PLOT_COLORS[color_key],
                    label=label, linewidth=PLOT_STYLE["linewidth"],
                    markersize=PLOT_STYLE["markersize"], capsize=PLOT_STYLE["capsize"])

    ax.set_xlabel("Transfer trial")
    ax.set_ylabel("P(4-key sequence was OLD)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Reuse of rewarded 4-key sequences during transfer")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    LOG.info("Saved OLD usage plot to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_old_sequence_breakdown(
    rows: Sequence[dict],
    max_trials: int | None,
    save_path: Path,
    show: bool,
) -> None:
    """Plot OLD_CORRECT_THIS vs OLD_CORRECT_OTHER usage per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    palette = [("OLD_CORRECT_THIS", PLOT_COLORS["valid"], "o"),
               ("OLD_CORRECT_OTHER", PLOT_COLORS["invalid"], "s")]

    for ax, condition, title in zip(axes, ["transfer_high", "transfer_low"], ["Transfer High", "Transfer Low"]):
        plotted = False
        for label, color, marker in palette:
            trials, means, sems = compute_condition_curve(
                rows, condition, lambda r, lbl=label: 1.0 if r["label"] == lbl else 0.0, max_trials,
            )
            if not trials:
                continue
            plotted = True
            ax.errorbar(trials, means, yerr=sems, fmt=f"{marker}-", color=color,
                        label=label.replace("_", " ").title(),
                        linewidth=PLOT_STYLE["linewidth"], markersize=PLOT_STYLE["markersize"],
                        capsize=PLOT_STYLE["capsize"])
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xlabel("Transfer trial")
        ax.set_title(title)

    axes[0].set_ylabel("P(sequence label)")
    axes[0].legend(frameon=False)
    fig.suptitle("Breakdown of OLD sequence usage", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    LOG.info("Saved OLD label breakdown plot to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_two_key_usage(
    rows: Sequence[dict],
    condition: str,
    max_trials: int | None,
    save_path: Path,
    show: bool,
) -> None:
    """Plot valid vs invalid 2-key usage for a single transfer condition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    mapping = [("valid_prev", "valid", "Valid"), ("invalid_matched", "invalid", "Invalid")]
    plotted = False

    for seq_type, color_key, label in mapping:
        trials, means, sems = compute_condition_curve(
            rows, condition, lambda r: float(r["count"]), max_trials,
            filter_fn=lambda r, st=seq_type: r["seq_type"] == st,
        )
        if not trials:
            continue
        plotted = True
        ax.errorbar(trials, means, yerr=sems,
                    fmt=f"{PLOT_MARKERS[color_key]}-", color=PLOT_COLORS[color_key],
                    linewidth=PLOT_STYLE["linewidth"], markersize=PLOT_STYLE["markersize"],
                    capsize=PLOT_STYLE["capsize"], label=label)

    if not plotted:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_xlabel("Transfer trial")
    ax.set_ylabel("Mean # of 2-key pairs")
    ax.set_title(f"2-key usage ({condition.replace('_', ' ')})")
    ax.set_ylim(-0.05, 2.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    LOG.info("Saved two-key usage plot for %s to %s", condition, save_path)
    if show:
        plt.show()
    plt.close(fig)


def _fit_mixed_with_fallback(
    rows: Sequence[dict],
    dv: str,
    formula_terms: list[str],
    participant_col: str = "participant_id",
):
    """Run MixedLM with OLS fallback for stability."""
    if not rows:
        raise ValueError("No data provided for regression")

    reg = MixedRegressor(rows, participant_col=participant_col)
    try:
        return reg.fit(dv, formula_terms)
    except Exception as exc:  # pragma: no cover - runtime fallback
        LOG.warning("Mixed model failed (%s); retrying with powell optimizer", exc)
    try:
        return reg.fit(dv, formula_terms, method="powell")
    except Exception as exc2:  # pragma: no cover - runtime fallback
        LOG.warning("Powell optimization failed (%s); falling back to clustered OLS", exc2)

    import pandas as pd  # type: ignore
    import statsmodels.formula.api as smf  # type: ignore

    df = pd.DataFrame(rows)
    rhs = " + ".join(formula_terms) if formula_terms else "1"
    model = smf.ols(f"{dv} ~ {rhs}", data=df)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df[participant_col]})
    return result, _create_summary_dataframe(result)


def fit_old_sequence_regression(rows: Sequence[dict], condition: str):
    """Fit OLD reuse model for a single transfer condition."""
    filtered = [dict(row) for row in rows if row.get("condition") == condition]
    if not filtered:
        raise ValueError(f"No rows available for condition {condition}")
    mean_trial = mean(row["trial_index"] for row in filtered)
    for row in filtered:
        row["trial_centered"] = row["trial_index"] - mean_trial
        row["old_flag"] = float(row["old_flag"])
    return _fit_mixed_with_fallback(filtered, "old_flag", ["trial_centered"])


def fit_two_key_regression(rows: Sequence[dict], condition: str):
    """Fit SeqType x Trial mixed model for 2-key counts."""
    filtered = [dict(row) for row in rows if row.get("condition") == condition]
    if not filtered:
        raise ValueError(f"No 2-key rows available for {condition}")

    mean_trial = mean(row["trial_index"] for row in filtered)
    for row in filtered:
        row["trial_centered"] = row["trial_index"] - mean_trial
        # Ensure indicator is float for statsmodels
        row["seq_type_indicator"] = float(row["seq_type_indicator"])
        row["count"] = float(row["count"])

    terms = ["seq_type_indicator", "trial_centered", "seq_type_indicator:trial_centered"]
    return _fit_mixed_with_fallback(filtered, "count", terms)


def run_experiments(args: argparse.Namespace) -> None:
    """Main orchestration for both experiments."""
    participants_high = load_participants(args.transfer_high)
    participants_low = load_participants(args.transfer_low)

    parsed_high = parse_condition(participants_high)
    parsed_low = parse_condition(participants_low)

    learning_memory_high = build_learning_memory(
        parsed_high["learning_sequences"],
        parsed_high["learning_correctness"],
        parsed_high["learning_goals"],
    )
    learning_memory_low = build_learning_memory(
        parsed_low["learning_sequences"],
        parsed_low["learning_correctness"],
        parsed_low["learning_goals"],
    )

    old_rows_high = label_transfer_sequences(
        parsed_high["transfer_sequences"],
        parsed_high["transfer_goals"],
        learning_memory_high,
        "transfer_high",
        args.max_transfer_trials,
    )
    old_rows_low = label_transfer_sequences(
        parsed_low["transfer_sequences"],
        parsed_low["transfer_goals"],
        learning_memory_low,
        "transfer_low",
        args.max_transfer_trials,
    )
    all_old_rows = old_rows_high + old_rows_low

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    high_tag = args.transfer_high.stem
    low_tag = args.transfer_low.stem

    old_prob_path = output_dir / f"old_sequence_probability_{high_tag}_{low_tag}.png"
    plot_old_sequence_probability(all_old_rows, args.max_transfer_trials, old_prob_path, args.show_plots)

    old_breakdown_path = output_dir / f"old_sequence_breakdown_{high_tag}_{low_tag}.png"
    plot_old_sequence_breakdown(all_old_rows, args.max_transfer_trials, old_breakdown_path, args.show_plots)

    if old_rows_high:
        _, summary_high = fit_old_sequence_regression(old_rows_high, "transfer_high")
        LOG.info("Experiment 1 (transfer_high) coefficients:\n%s", summary_high.to_string(index=False))
    else:
        LOG.warning("No transfer_high rows available for Experiment 1 regression; skipping.")

    if old_rows_low:
        _, summary_low = fit_old_sequence_regression(old_rows_low, "transfer_low")
        LOG.info("Experiment 1 (transfer_low) coefficients:\n%s", summary_low.to_string(index=False))
    else:
        LOG.warning("No transfer_low rows available for Experiment 1 regression; skipping.")

    # Experiment 2
    rules = StarMakingRules()
    valid_pairs = set(rules.learning_rules["low"].keys())

    two_key_rows_high = count_two_key_usage(
        parsed_high["transfer_sequences"],
        "transfer_high",
        valid_pairs,
        args.max_transfer_trials,
    )
    two_key_rows_low = count_two_key_usage(
        parsed_low["transfer_sequences"],
        "transfer_low",
        valid_pairs,
        args.max_transfer_trials,
    )

    high_two_key_path = output_dir / f"two_key_usage_transfer_high_{high_tag}.png"
    low_two_key_path = output_dir / f"two_key_usage_transfer_low_{low_tag}.png"
    plot_two_key_usage(two_key_rows_high, "transfer_high", args.max_transfer_trials, high_two_key_path, args.show_plots)
    plot_two_key_usage(two_key_rows_low, "transfer_low", args.max_transfer_trials, low_two_key_path, args.show_plots)

    if two_key_rows_high:
        _, two_key_summary_high = fit_two_key_regression(two_key_rows_high, "transfer_high")
        LOG.info("Experiment 2 (transfer_high) coefficients:\n%s", two_key_summary_high.to_string(index=False))
    else:
        LOG.warning("No transfer_high rows available for Experiment 2 regression; skipping.")

    if two_key_rows_low:
        _, two_key_summary_low = fit_two_key_regression(two_key_rows_low, "transfer_low")
        LOG.info("Experiment 2 (transfer_low) coefficients:\n%s", two_key_summary_low.to_string(index=False))
    else:
        LOG.warning("No transfer_low rows available for Experiment 2 regression; skipping.")

    LOG.info(
        "Completed analyses. Figures saved to %s. Consider archiving model summaries separately if needed.",
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze transfer-phase sequence reuse and 2-key usage.")
    parser.add_argument("--transfer-high", type=Path, required=True, help="Path to transfer_high *_experiment.json")
    parser.add_argument("--transfer-low", type=Path, required=True, help="Path to transfer_low *_experiment.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots/sequence_item_transfer_usage"),
        help="Directory for saved plots",
    )
    parser.add_argument(
        "--max-transfer-trials",
        type=int,
        default=40,
        help="Number of transfer trials to include (use <= total transfer trials)",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively in addition to saving.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()

"""
Sample usage:
    # Run for item condition
    PYTHONPATH=. python analysis/new_experiments/sequence_item_transfer_usage.py \
        --transfer-high results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json \
        --transfer-low results/gpt-oss-120b/transfer_low/item_20251122_040528_experiment.json \
        --output-dir results/gpt-oss-120b/experiments/sequence_transfer_usage/item

    # Run for no_item condition
    PYTHONPATH=. python analysis/new_experiments/sequence_item_transfer_usage.py \
        --transfer-high results/gpt-oss-120b/transfer_high/no_item_20251122_075552_experiment.json \
        --transfer-low results/gpt-oss-120b/transfer_low/no_item_20251122_040337_experiment.json \
        --output-dir results/gpt-oss-120b/experiments/sequence_transfer_usage/no_item

    # With max trials limit and interactive display
    PYTHONPATH=. python analysis/new_experiments/sequence_item_transfer_usage.py \
        --transfer-high path/to/high.json \
        --transfer-low path/to/low.json \
        --output-dir output/ \
        --max-trials 30 \
        --show
"""
