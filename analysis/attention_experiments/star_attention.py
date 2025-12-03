"""
Here we aim to determine which star attention is focused most on during transfer phase:

Given path to attention analysis: for example "/scratch/gpfs/ABDIENG/bw5889/code/learning/Hierarchy_LLM/hierarchy_action_sequence/results/gpt-oss-120b/attention_analysis/20251201_215244_attention_analysis.json"
For each participant we want to collect the attention probabilities of "Star_i", where i \in {0,1,2,3,4,5}
Tricky:
- In tokenization "Star_i" is tokenized as ["Star", "_", "i"]
- Good thing is that these 3 tokens are consequtive in the tokenized sequence, thus we characterize attention prob to "Star_i" as sum of attention probs to these 3 tokens when they are consequtive
- For each participant we sum the attention probs to "Star_i" across all conversation history.
- Then for "Star_i" we average the attention probs across all participants
- Finally we plot a bar plot of average attention probs to each "Star_i" and save to an output dir, print also the numerical values
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.functions import setup_plot_style


def extract_star_attention(attention_weights: Dict[str, List]) -> Dict[str, float]:
    """
    Extract attention probabilities for each Star_i from a participant's attention weights.

    Args:
        attention_weights: Dict mapping token_index (str) -> [token_text, probability]

    Returns:
        Dict mapping "Star_0" through "Star_5" to their total attention probabilities
    """
    # Initialize star attention sums
    star_attention = {f"Star_{i}": 0.0 for i in range(6)}

    # Convert to sorted list by token index
    token_indices = sorted([(int(idx), data) for idx, data in attention_weights.items()])

    # Scan for consecutive "Star", "_", "digit" triplets
    for i in range(len(token_indices) - 2):
        token1_idx, (token1_text, prob1) = token_indices[i]
        token2_idx, (token2_text, prob2) = token_indices[i + 1]
        token3_idx, (token3_text, prob3) = token_indices[i + 2]

        # Check if tokens are consecutive and match "Star_i" pattern
        if (token2_idx == token1_idx + 1 and token3_idx == token2_idx + 1 and
            token1_text == "Star" and token2_text == "_" and token3_text in "012345"):

            star_key = f"Star_{token3_text}"
            star_attention[star_key] += prob1 + prob2 + prob3

    return star_attention


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention probabilities for Star_0 through Star_5"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to attention analysis JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: star_attention_output/ next to input file)"
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "star_attention_output"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "csv").mkdir(exist_ok=True)

    # Load attention analysis data
    print(f"Loading attention analysis from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Extract star attention for each participant
    participant_star_attention = []

    for result in data["replay_results"]:
        participant_id = result["participant_id"]
        attention_weights = result["attention_weights"]

        star_attention = extract_star_attention(attention_weights)
        star_attention["participant_id"] = participant_id
        participant_star_attention.append(star_attention)

    # Create DataFrame
    df = pd.DataFrame(participant_star_attention)
    df = df.set_index("participant_id")

    # Calculate average across participants
    star_columns = [f"Star_{i}" for i in range(6)]
    avg_attention = df[star_columns].mean()
    std_attention = df[star_columns].std()

    # Filter out stars with zero attention
    nonzero_mask = avg_attention > 0
    nonzero_stars = avg_attention[nonzero_mask].index.tolist()
    nonzero_avg = avg_attention[nonzero_mask]
    nonzero_std = std_attention[nonzero_mask]

    # Print numerical values (only non-zero)
    print("\n" + "=" * 60)
    print("STAR ATTENTION PROBABILITIES (Average across participants)")
    print("=" * 60)
    for star in nonzero_stars:
        print(f"{star}: {avg_attention[star]:.6f} ± {std_attention[star]:.6f}")
    print("=" * 60)
    if len(nonzero_stars) < len(star_columns):
        excluded = [s for s in star_columns if s not in nonzero_stars]
        print(f"Note: Excluded stars with zero attention: {', '.join(excluded)}")
    print()

    # Save detailed results to CSV (only non-zero)
    csv_path = output_dir / "csv" / "star_attention_summary.csv"
    summary_df = pd.DataFrame({
        "Star": nonzero_stars,
        "Mean_Attention": nonzero_avg.values,
        "Std_Attention": nonzero_std.values
    })
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to: {csv_path}")

    # Save per-participant data
    participant_csv_path = output_dir / "csv" / "star_attention_per_participant.csv"
    df.to_csv(participant_csv_path)
    print(f"Saved per-participant data to: {participant_csv_path}")

    # Create bar plot (only non-zero)
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(nonzero_stars))
    bars = ax.bar(x_pos, nonzero_avg.values, yerr=nonzero_std.values,
                   capsize=5, alpha=0.8, color='#1d3557', edgecolor='black', linewidth=1.2)

    ax.set_xlabel("Star", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Attention Probability", fontsize=14, fontweight='bold')
    ax.set_title("Attention Probabilities for Stars (Transfer Phase)", fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nonzero_stars, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "plots" / "star_attention_probabilities.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")

    plt.close()

    print(f"\nAnalysis complete! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
