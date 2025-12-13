"""Lightweight analysis helpers for action-sequence experiments."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

# =============================================================================
# Shared Plot Styling
# =============================================================================
PLOT_COLORS = {
    "invalid": "#e63946",      # red
    "valid": "#1d3557",        # blue
    "high": "#E15759",         # red (transfer high)
    "low": "#4EBFD8",          # cyan (transfer low)
}
PLOT_MARKERS = {"valid": "o", "invalid": "s", "high": "o", "low": "s"}
PLOT_STYLE = {
    "linewidth": 2.2,
    "markersize": 7,
    "capsize": 4,
    "alpha": 0.9,
    "fill_alpha": 0.15,
}


def setup_plot_style():
    """Apply consistent matplotlib/seaborn styling."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.star_making.assets_utils import ACTION_ENCODE_1, ACTION_ENCODE_2

LOG = logging.getLogger(__name__)
_ACTIONS_RE = re.compile(r"Actions:\s*\[([^\]]+)\]", re.IGNORECASE)
EMPTY_TRIAL_RESULT = {"learning_trials": {}, "transfer_trials": {}}


def _blocks(items: Sequence, size: int = 10) -> list[list]:
    """Chunk a sequence into consecutive blocks."""
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _flatten_blocks(blocks: Iterable[Iterable]) -> list:
    """Concatenate block-structured data while preserving order."""
    flat: list = []
    for block in blocks:
        flat.extend(list(block))
    return flat


def _extract_actions_from_text(content: str) -> list[str] | None:
    """Pull the four-action sequence from a user message."""
    if "\n\n" not in content:
        return None

    match = _ACTIONS_RE.search(content)
    if not match:
        return None

    actions = re.findall(r"[UIOP]", match.group(1).upper())
    if len(actions) != 4:
        return None
    return actions


def _split_and_block_trials(
    participants: Sequence[dict],
    data_extractor: Callable[[dict, int], list],
    data_name: str = "data",
) -> dict[str, dict[int, list[list]]]:
    """Generic trial splitter for learning/transfer cutoff.

    Args:
        participants: Participant data dictionaries.
        data_extractor: Function that extracts trial data from (participant, position).
        data_name: Description for logging purposes.

    Returns:
        Dict with 'learning_trials' and 'transfer_trials' keys.
    """
    if not participants:
        return EMPTY_TRIAL_RESULT

    learning: dict[int, list[list]] = {}
    transfer: dict[int, list[list]] = {}

    for position, participant in enumerate(participants):
        participant_idx = participant.get("participant_id", position)
        data = data_extractor(participant, position)

        if not data:
            LOG.warning("No %s found for participant %s", data_name, participant_idx)
            continue

        cutoff = (2 * len(data)) // 3
        learning[participant_idx] = _blocks(data[:cutoff])
        transfer[participant_idx] = _blocks(data[cutoff:])

    return {"learning_trials": learning, "transfer_trials": transfer}


def parse_action_sequence(participants: Sequence[dict]) -> dict[str, dict[int, list[list[list[int]]]]]:
    """Parse encoded action sequences and bucket them into blocks of 10 per participant.

    Args:
        participants: Iterable of participant dicts loaded from an experiment JSON.

    Returns:
        Dict with ``learning_trials`` and ``transfer_trials`` keyed by participant idx,
        where each value is a list of blocks (each block up to 10 sequences) ordered
        from the start of the conversation history.
    """
    split = len(participants) // 2 if participants else 0

    def extract_sequences(participant: dict, position: int) -> list[list[int]]:
        encoder = ACTION_ENCODE_1 if position < split else ACTION_ENCODE_2
        sequences: list[list[int]] = []
        for message in participant.get("conversation_history", []):
            if message.get("role") == "user":
                if actions := _extract_actions_from_text(message.get("content", "")):
                    sequences.append([encoder[action] for action in actions])
        return sequences

    return _split_and_block_trials(participants, extract_sequences, "action sequences")


def parse_trial_correctness(participants: Sequence[dict]) -> dict[str, dict[int, list[list[bool]]]]:
    """Bucket trial correctness into blocks of 10 for each participant."""

    def extract_correctness(participant: dict, position: int) -> list[bool]:
        return [bool(trial.get("success")) for trial in participant.get("trials", [])]

    return _split_and_block_trials(participants, extract_correctness, "trial correctness")


def parse_goal_stars(participants: Sequence[dict]) -> dict[str, dict[int, list[list[str | None]]]]:
    """Bucket goal-star labels into blocks of 10 for each participant."""

    def extract_goal_stars(participant: dict, position: int) -> list[str | None]:
        return [trial.get("goal_star") for trial in participant.get("trials", [])]

    return _split_and_block_trials(participants, extract_goal_stars, "goal stars")


def _create_summary_dataframe(result):
    """Create summary dataframe from statsmodels result."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Summary dataframe requires pandas; install pandas>=2.0.") from exc

    conf = result.conf_int()
    return pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "z": result.tvalues,
            "p": result.pvalues,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
        }
    )


class MixedRegressor:
    """Convenience wrapper around statsmodels mixed-effects regression."""

    def __init__(self, trials: Iterable[dict], participant_col: str = "participant_id") -> None:
        self.participant_col = participant_col
        self._pd = self._lazy_import_pandas()
        self._smf = self._lazy_import_statsmodels()
        self.data = self._pd.DataFrame(trials)
        if participant_col not in self.data.columns:
            raise ValueError(f"trials must include '{participant_col}' for grouping")

    @staticmethod
    def _lazy_import_pandas():
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError("MixedRegressor requires pandas; install pandas>=2.0.") from exc
        return pd

    @staticmethod
    def _lazy_import_statsmodels():
        try:
            import statsmodels.formula.api as smf  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError("MixedRegressor requires statsmodels; install statsmodels>=0.14.") from exc
        return smf

    def fit(
        self,
        dv: str,
        fixed_effects: Sequence[str] | str,
        subset: str | Callable[[object], object] | None = None,
        re_formula: str = "1",
        method: str = "lbfgs",
        fit_kwargs: dict | None = None,
    ):
        """Fit a mixed-effects model with random intercepts per participant.

        Args:
            dv: Dependent variable column name.
            fixed_effects: Effects to include (list or a patsy-style string).
            subset: Optional query string or callable mask to filter rows.
            re_formula: Random-effects formula (defaults to intercept only).
            method: Optimizer passed to statsmodels (default: lbfgs).
            fit_kwargs: Optional kwargs forwarded to statsmodels MixedLM.fit.

        Returns:
            Tuple of (statsmodels result, tidy summary dataframe).
        """
        df = self.data
        if subset is not None:
            df = df.query(subset) if isinstance(subset, str) else df[subset(df)]
        if df.empty:
            raise ValueError("No data to fit after applying subset filter")

        fixed = fixed_effects if isinstance(fixed_effects, str) else " + ".join(fixed_effects) or "1"
        formula = f"{dv} ~ {fixed}"

        model = self._smf.mixedlm(formula, data=df, groups=df[self.participant_col], re_formula=re_formula)
        fit_kwargs = fit_kwargs or {}
        result = model.fit(reml=False, method=method, disp=False, **fit_kwargs)

        return result, _create_summary_dataframe(result)


def main() -> None:
    """Simple CLI to parse a results JSON and echo trial counts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    default_path = Path("results") / "gpt-oss-120b" / "transfer_low" / "20251119_162228_experiment.json"
    parser = argparse.ArgumentParser(description="Parse action sequences from experiment logs.")
    parser.add_argument("input_path", nargs="?", default=str(default_path), help="Path to *_experiment.json")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = parse_action_sequence(data.get("participants", []))
    LOG.info("Loaded %s participants from %s", len(data.get("participants", [])), input_path)
    LOG.info(
        "Parsed sequences: %s learning keys, %s transfer keys",
        len(parsed["learning_trials"]),
        len(parsed["transfer_trials"]),
    )


if __name__ == "__main__":
    main()
