#!/usr/bin/env python3
"""
Analysis script for old format inference batch JSON results.

This script analyzes participant performance in the star-making game,
calculating success rates, learning curves, and generating visualizations.
It handles the old format where trial data is stored in conversation_history.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.dpi'] = 150
PERCENT_FORMATTER = FuncFormatter(lambda y, _: f"{y:.0%}")
HEATMAP_CMAP = sns.color_palette(['#f8d7da', '#1f7a8c'], as_cmap=True)


def _gradient_colors(n: int, palette: str = "viridis") -> List:
    """Return a smooth gradient with at least three colors."""
    return sns.color_palette(palette, max(n, 3))


def load_json_data(file_path: str) -> Dict:
    """
    Load JSON data from file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the parsed JSON data
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded data for {len(data['participants'])} participants")
    return data


def parse_conversation_history(conversation_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Parse conversation history to extract trial data.

    Args:
        conversation_history: List of conversation messages with role and content

    Returns:
        List of trial dictionaries with trial number, goal_star, and success
    """
    trials = []
    current_trial = None
    trial_number = 0

    # Pattern to extract goal star from state messages
    goal_star_pattern = re.compile(r'Goal star:\s*(Star_\d+)', re.IGNORECASE)

    # Pattern to detect trial start (fresh state)
    trial_start_pattern = re.compile(r'Current state:\s*\nGoal star:.*\nItems:\s*\[None,\s*None\]\.\s*\nActions:\s*\[\]', re.DOTALL)

    # Pattern to detect success
    success_pattern = re.compile(r'Congratulations!.*successfully made.*goal star', re.IGNORECASE)

    # Pattern to detect failure
    failure_pattern = re.compile(r'Trial ended\..*did not make.*goal star', re.IGNORECASE)

    for i, message in enumerate(conversation_history):
        if message['role'] != 'user':
            continue

        content = message['content']

        # Check if this is a trial start
        if trial_start_pattern.search(content):
            # If we have a previous trial that hasn't been closed, it failed
            if current_trial is not None:
                current_trial['success'] = False
                trials.append(current_trial)
                trial_number += 1

            # Extract goal star
            goal_match = goal_star_pattern.search(content)
            goal_star = goal_match.group(1) if goal_match else 'Unknown'

            # Start new trial
            current_trial = {
                'trial': trial_number,
                'goal_star': goal_star,
                'success': None  # Will be determined later
            }

        # Check for success or failure messages
        elif current_trial is not None:
            if success_pattern.search(content):
                current_trial['success'] = True
                trials.append(current_trial)
                current_trial = None
                trial_number += 1
            elif failure_pattern.search(content):
                current_trial['success'] = False
                trials.append(current_trial)
                current_trial = None
                trial_number += 1

    # Handle last trial if not closed
    if current_trial is not None:
        # Default to failure if trial wasn't explicitly succeeded
        current_trial['success'] = False
        trials.append(current_trial)

    return trials


def _resolve_participant_id(participant: Dict[str, Any], fallback: int) -> Any:
    """
    Determine the participant identifier across schema variations.
    """
    for key in ('participant_id', 'participant', 'id', 'name'):
        if participant.get(key) is not None:
            return participant[key]
    return fallback


def extract_participant_data(data: Dict) -> pd.DataFrame:
    """
    Extract participant-level summary data.

    Supports both formats:
    - New format: trials data directly in 'trials' key
    - Old format: trials data parsed from conversation_history

    Args:
        data: Parsed JSON data

    Returns:
        DataFrame with columns: participant_id, total_trials, successful_trials, success_rate
    """
    participants_summary = []

    for idx, participant in enumerate(data['participants']):
        participant_id = _resolve_participant_id(participant, idx)

        # Check for new format with direct trials data
        trials = participant.get('trials', [])

        # Fall back to parsing conversation history if trials not available
        if not trials:
            conversation_history = participant.get('conversation_history', [])
            trials = parse_conversation_history(conversation_history)

        # Calculate from parsed trials
        computed_total = len(trials)
        computed_successes = sum(1 for trial in trials if trial.get('success'))

        # Use summary if available, otherwise use computed values
        summary = participant.get('summary')
        summary = summary if isinstance(summary, dict) else {}

        total_trials = summary.get('total_trials', computed_total)
        if total_trials is None:
            total_trials = computed_total
        successful_trials = summary.get('successful_trials')
        if successful_trials is None:
            successful_trials = computed_successes

        try:
            total_trials = int(total_trials)
        except (TypeError, ValueError):
            total_trials = computed_total
        try:
            successful_trials = int(successful_trials)
        except (TypeError, ValueError):
            successful_trials = computed_successes

        success_rate = summary.get('success_rate')
        if success_rate is None:
            success_rate = participant.get('success_rate')
        if success_rate is None:
            success_rate = (successful_trials / total_trials) if total_trials else 0.0
        try:
            success_rate = float(success_rate)
        except (TypeError, ValueError):
            success_rate = (successful_trials / total_trials) if total_trials else 0.0

        participants_summary.append({
            'participant_id': participant_id,
            'total_trials': int(total_trials),
            'successful_trials': int(successful_trials),
            'success_rate': float(success_rate)
        })

    return pd.DataFrame(participants_summary)


def extract_trial_details(data: Dict) -> pd.DataFrame:
    """
    Extract detailed trial-by-trial data for all participants.

    Supports both formats:
    - New format: trials data directly in 'trials' key
    - Old format: trials data parsed from conversation_history

    Args:
        data: Parsed JSON data

    Returns:
        DataFrame with columns: participant_id, trial, goal_star, success
    """
    trial_records = []

    for p_idx, participant in enumerate(data['participants']):
        participant_id = _resolve_participant_id(participant, p_idx)

        # Check for new format with direct trials data
        trials = participant.get('trials', [])

        # Fall back to parsing conversation history if trials not available
        if not trials:
            conversation_history = participant.get('conversation_history', [])
            trials = parse_conversation_history(conversation_history)

        for trial in trials:
            trial_records.append({
                'participant_id': participant_id,
                'trial': trial['trial'],
                'goal_star': trial['goal_star'],
                'success': trial['success']
            })

    return pd.DataFrame(trial_records)


def _format_goal_star_label(config: Dict[str, Any],
                            experiment_type: str,
                            stars: List[str] = None) -> str:
    """
    Create a readable label for goal star usage in titles/summary output.
    """
    experiment_cfg = config.get('experiment', {})
    specified = experiment_cfg.get('specify_star')
    if specified:
        return specified
    if experiment_type == 'multi':
        if stars:
            return f"Multiple Stars ({', '.join(stars)})"
        return 'Multiple Stars'
    return 'Unspecified'


def detect_experiment_type(trial_df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Detect if experiment is single-star or multi-star.

    Args:
        trial_df: DataFrame with trial details

    Returns:
        Tuple of (experiment_type, list_of_stars)
        experiment_type is either 'single' or 'multi'
    """
    unique_stars = sorted(trial_df['goal_star'].unique())
    experiment_type = 'single' if len(unique_stars) == 1 else 'multi'
    return experiment_type, unique_stars


def calculate_learning_curves(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate learning curves showing success rates over trials.

    Args:
        trial_df: DataFrame with trial details

    Returns:
        DataFrame with columns: trial_number, cumulative_success_rate,
                                individual_trial_success_rate, num_participants_succeeded
    """
    learning_data = []

    # Group by trial number
    for trial_num in sorted(trial_df['trial'].unique()):
        # Get all attempts at this trial across participants
        trial_attempts = trial_df[trial_df['trial'] == trial_num]

        # Individual trial success rate (for this specific trial)
        individual_success_rate = trial_attempts['success'].mean()
        num_succeeded = trial_attempts['success'].sum()

        # Cumulative success rate (from trial 0 to current trial)
        cumulative_attempts = trial_df[trial_df['trial'] <= trial_num]
        cumulative_success_rate = cumulative_attempts['success'].mean()

        learning_data.append({
            'trial_number': trial_num,
            'cumulative_success_rate': cumulative_success_rate,
            'individual_trial_success_rate': individual_success_rate,
            'num_participants_succeeded': int(num_succeeded),
            'total_participants': len(trial_attempts)
        })

    return pd.DataFrame(learning_data)


def create_success_matrix(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a participant × trial success matrix.

    Args:
        trial_df: DataFrame with trial details

    Returns:
        DataFrame where rows are participants, columns are trials, values are success (0/1)
    """
    pivot_df = trial_df.pivot(
        index='participant_id',
        columns='trial',
        values='success'
    ).fillna(0)
    # Convert boolean to int for better visualization
    return pivot_df.astype(int)


def save_csv_outputs(participant_df: pd.DataFrame,
                     learning_df: pd.DataFrame,
                     trial_df: pd.DataFrame,
                     output_dir: Path,
                     experiment_type: str = 'single',
                     stars: List[str] = None) -> None:
    """
    Save analysis results to CSV files.

    Args:
        participant_df: Participant summary data
        learning_df: Learning curve data
        trial_df: Detailed trial data
        output_dir: Directory to save CSV files
        experiment_type: 'single' or 'multi'
        stars: List of stars (for multi-star experiments)
    """
    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregate participant summary
    participant_file = csv_dir / 'participant_summary.csv'
    participant_df.to_csv(participant_file, index=False)
    print(f"✓ Saved participant summary to {participant_file}")

    # Save aggregate learning curve
    learning_file = csv_dir / 'learning_curve.csv'
    learning_df.to_csv(learning_file, index=False)
    print(f"✓ Saved learning curve to {learning_file}")

    # Save trial details
    trial_file = csv_dir / 'trial_details.csv'
    trial_df.to_csv(trial_file, index=False)
    print(f"✓ Saved trial details to {trial_file}")


def create_visualizations(participant_df: pd.DataFrame,
                         learning_df: pd.DataFrame,
                         success_matrix: pd.DataFrame,
                         trial_df: pd.DataFrame,
                         config: Dict,
                         output_dir: Path,
                         experiment_type: str = 'single',
                         stars: List[str] = None,
                         show_lines: bool = True,
                         start_trial: int = 0,
                         end_trial: int = None,
                         transfer_start: int = 80) -> None:
    """
    Create and save visualization plots.

    Args:
        participant_df: Participant summary data
        learning_df: Learning curve data
        success_matrix: Participant × trial success matrix
        trial_df: DataFrame with trial details
        config: Experiment configuration
        output_dir: Directory to save plots
        experiment_type: 'single' or 'multi'
        stars: List of stars (for multi-star experiments)
        show_lines: Whether to show connecting lines in learning curves
        start_trial: Start trial for filtering
        end_trial: End trial for filtering
        transfer_start: Trial index where transfer phase starts (default: 80)
    """
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    percent_formatter = PERCENT_FORMATTER

    # Extract experiment info for titles
    model_cfg = config.get('model', {})
    model_family = model_cfg.get('family', 'UnknownModel')
    model_size = model_cfg.get('size', 'UnknownSize')
    model_name = f"{model_family}-{model_size}"
    goal_star = _format_goal_star_label(config, experiment_type, stars)

    # Add trial range info to titles if filtering trials
    if start_trial > 0 and end_trial is not None:
        trial_range_suffix = f" (Trials {start_trial}-{end_trial})"
    elif start_trial > 0:
        trial_range_suffix = f" (Trials {start_trial}+)"
    elif end_trial is not None:
        trial_range_suffix = f" (Trials 0-{end_trial})"
    else:
        trial_range_suffix = ""

    # 1. Bar chart: Success rates by participant
    participant_plot_df = participant_df.sort_values('success_rate', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    bar_positions = np.arange(len(participant_plot_df))
    bar_colors = _gradient_colors(len(participant_plot_df), 'viridis')
    bars = ax.bar(bar_positions,
                  participant_plot_df['success_rate'],
                  color=bar_colors,
                  edgecolor='none',
                  alpha=0.9)
    ax.set_xlabel('Participant ID', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'Success Rate by Participant{trial_range_suffix}\n({model_name}, {goal_star})', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([str(pid) for pid in participant_plot_df['participant_id']], rotation=30, ha='right')
    ax.yaxis.set_major_formatter(percent_formatter)
    mean_rate = participant_plot_df['success_rate'].mean()
    ax.axhline(mean_rate, color='#6c757d', linestyle='--', linewidth=1, label=f'Mean: {mean_rate:.0%}')
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    sns.despine(ax=ax)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.0%}',
                ha='center', va='bottom', fontsize=9, color='#1b1b1b')

    plt.tight_layout()
    participant_plot = plots_dir / 'participant_success_rates.png'
    plt.savefig(participant_plot, bbox_inches='tight')
    print(f"✓ Saved participant success rates plot to {participant_plot}")
    plt.close()

    # 2a. Line plot: Cumulative Success Rate Learning Curve
    fig, ax = plt.subplots(figsize=(12, 6))

    linestyle = '-' if show_lines else 'none'
    linewidth = 2 if show_lines else 0
    marker_size = 5 if show_lines else 7

    ax.plot(learning_df['trial_number'],
            learning_df['cumulative_success_rate'],
            marker='o', linestyle=linestyle, linewidth=linewidth, markersize=marker_size,
            label='Cumulative Success Rate', color='#238a8d')
    ax.fill_between(learning_df['trial_number'],
                    learning_df['cumulative_success_rate'],
                    alpha=0.15, color='#238a8d')

    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    title_suffix = f'({model_name}, Multi-Star)' if experiment_type == 'multi' else f'({model_name}, {goal_star})'
    ax.set_title(f'Learning Curve: Cumulative Success Rate Over Trials{trial_range_suffix}\n{title_suffix}',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 0.6])
    ax.yaxis.set_major_formatter(percent_formatter)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    sns.despine(ax=ax)

    plt.tight_layout()
    learning_plot_cumulative = plots_dir / 'learning_curves_cumulative.png'
    plt.savefig(learning_plot_cumulative, bbox_inches='tight')
    print(f"✓ Saved cumulative learning curve plot to {learning_plot_cumulative}")
    plt.close()

    # 2b. Line plot: Per-Trial Success Rate Learning Curve
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(learning_df['trial_number'],
            learning_df['individual_trial_success_rate'],
            marker='s', linestyle=linestyle, linewidth=linewidth, markersize=marker_size,
            label='Per-Trial Success Rate', color='#f06f5c', alpha=0.9)
    ax.fill_between(learning_df['trial_number'],
                    learning_df['individual_trial_success_rate'],
                    alpha=0.12, color='#f06f5c')

    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    title_suffix = f'({model_name}, Multi-Star)' if experiment_type == 'multi' else f'({model_name}, {goal_star})'
    ax.set_title(f'Learning Curve: Per-Trial Success Rate Over Trials{trial_range_suffix}\n{title_suffix}',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 0.6])
    ax.yaxis.set_major_formatter(percent_formatter)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    sns.despine(ax=ax)

    plt.tight_layout()
    learning_plot_per_trial = plots_dir / 'learning_curves_per_trial.png'
    plt.savefig(learning_plot_per_trial, bbox_inches='tight')
    print(f"✓ Saved per-trial learning curve plot to {learning_plot_per_trial}")
    plt.close()

    # 3. Heatmap: Participant × Trial success matrix
    ordered_matrix = success_matrix.sort_index().sort_index(axis=1)
    heatmap_width = max(12, 0.4 * ordered_matrix.shape[1])
    heatmap_height = max(5, 0.4 * ordered_matrix.shape[0])
    fig, ax = plt.subplots(figsize=(heatmap_width, heatmap_height))

    sns.heatmap(ordered_matrix,
                cmap=HEATMAP_CMAP,
                cbar_kws={'label': 'Success', 'ticks': [0, 1]},
                linewidths=0.3, linecolor='white',
                ax=ax, vmin=0, vmax=1)

    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Participant ID', fontsize=12)
    title_suffix = f'({model_name}, Multi-Star)' if experiment_type == 'multi' else f'({model_name}, {goal_star})'
    ax.set_title(f'Success Matrix: Participant × Trial{trial_range_suffix}\n{title_suffix}',
                 fontsize=14, fontweight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    heatmap_plot = plots_dir / 'success_heatmap.png'
    plt.savefig(heatmap_plot, bbox_inches='tight')
    print(f"✓ Saved success heatmap to {heatmap_plot}")
    plt.close()

    # 4. Box plot: Distribution of success rates
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=participant_df['success_rate'], ax=ax, width=0.4,
                color='#bcd8f4', fliersize=0)
    sns.stripplot(y=participant_df['success_rate'], ax=ax,
                  color='#1b4965', size=6, alpha=0.7, jitter=0.08)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_xticks([])
    ax.set_title(f'Distribution of Success Rates Across Participants{trial_range_suffix}\n({model_name}, {goal_star})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.yaxis.set_major_formatter(percent_formatter)
    sns.despine(ax=ax, left=False, bottom=True)

    stats_text = f"Mean: {participant_df['success_rate'].mean():.0%}\n"
    stats_text += f"Median: {participant_df['success_rate'].median():.0%}\n"
    stats_text += f"Std: {participant_df['success_rate'].std():.1%}"
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    boxplot_file = plots_dir / 'success_rate_distribution.png'
    plt.savefig(boxplot_file, bbox_inches='tight')
    print(f"✓ Saved success rate distribution plot to {boxplot_file}")
    plt.close()

    # 5. Two-subplot figure: Learning phase vs Transfer phase per-trial success rates
    # Split learning_df into learning and transfer phases based on transfer_start
    learning_phase_df = learning_df[learning_df['trial_number'] < transfer_start]
    transfer_phase_df = learning_df[learning_df['trial_number'] >= transfer_start]

    # Only create this plot if we have data in both phases
    if len(learning_phase_df) > 0 and len(transfer_phase_df) > 0:
        # Scale subplot widths to reflect the number of trials in each phase
        learning_span = int(learning_phase_df['trial_number'].max() - learning_phase_df['trial_number'].min() + 1)
        transfer_span = int(transfer_phase_df['trial_number'].max() - transfer_phase_df['trial_number'].min() + 1)
        fig, (ax_learn, ax_transfer) = plt.subplots(
            1, 2, figsize=(14, 5),
            gridspec_kw={'width_ratios': [learning_span, transfer_span]}
        )

        # Common styling
        learning_color = '#2ecc71'  # Green for learning
        transfer_color = '#e74c3c'  # Red for transfer

        # Left subplot: Learning phase
        ax_learn.plot(learning_phase_df['trial_number'],
                      learning_phase_df['individual_trial_success_rate'],
                      marker='o', linestyle=linestyle, linewidth=linewidth, markersize=marker_size,
                      label='Per-Trial Success Rate', color=learning_color)
        ax_learn.fill_between(learning_phase_df['trial_number'],
                              learning_phase_df['individual_trial_success_rate'],
                              alpha=0.15, color=learning_color)

        ax_learn.set_xlabel('Trial Number', fontsize=12)
        ax_learn.set_ylabel('Success Rate', fontsize=12)
        ax_learn.set_title(f'Learning Phase (Trials 0-{transfer_start - 1})', fontsize=13, fontweight='bold')
        ax_learn.set_ylim([0, 0.6])
        ax_learn.yaxis.set_major_formatter(percent_formatter)
        ax_learn.xaxis.set_major_locator(MultipleLocator(10))
        sns.despine(ax=ax_learn)

        # Add mean line and stats for learning phase
        learn_mean = learning_phase_df['individual_trial_success_rate'].mean()
        ax_learn.axhline(learn_mean, color=learning_color, linestyle='--', linewidth=1.5, alpha=0.7)
        ax_learn.text(0.97, 0.97, f"Mean: {learn_mean:.1%}",
                      transform=ax_learn.transAxes, verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

        # Right subplot: Transfer phase
        ax_transfer.plot(transfer_phase_df['trial_number'],
                         transfer_phase_df['individual_trial_success_rate'],
                         marker='s', linestyle=linestyle, linewidth=linewidth, markersize=marker_size,
                         label='Per-Trial Success Rate', color=transfer_color)
        ax_transfer.fill_between(transfer_phase_df['trial_number'],
                                 transfer_phase_df['individual_trial_success_rate'],
                                 alpha=0.15, color=transfer_color)

        ax_transfer.set_xlabel('Trial Number', fontsize=12)
        ax_transfer.set_ylabel('Success Rate', fontsize=12)
        max_trial = int(transfer_phase_df['trial_number'].max())
        ax_transfer.set_title(f'Transfer Phase (Trials {transfer_start}-{max_trial})', fontsize=13, fontweight='bold')
        ax_transfer.set_ylim([0, 0.6])
        ax_transfer.yaxis.set_major_formatter(percent_formatter)
        ax_transfer.xaxis.set_major_locator(MultipleLocator(10))
        sns.despine(ax=ax_transfer)

        # Add mean line and stats for transfer phase
        transfer_mean = transfer_phase_df['individual_trial_success_rate'].mean()
        ax_transfer.axhline(transfer_mean, color=transfer_color, linestyle='--', linewidth=1.5, alpha=0.7)
        ax_transfer.text(0.97, 0.97, f"Mean: {transfer_mean:.1%}",
                         transform=ax_transfer.transAxes, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

        # Main title
        fig.suptitle(f'Per-Trial Success Rate: Learning vs Transfer Phase\n({model_name}, {goal_star})',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        phase_comparison_file = plots_dir / 'learning_vs_transfer_per_trial.png'
        plt.savefig(phase_comparison_file, bbox_inches='tight')
        print(f"✓ Saved learning vs transfer phase plot to {phase_comparison_file}")
        plt.close()
    else:
        # Log a message if we can't create the plot
        if len(learning_phase_df) == 0:
            print(f"⚠ Skipped learning vs transfer plot: no trials before transfer_start={transfer_start}")
        elif len(transfer_phase_df) == 0:
            print(f"⚠ Skipped learning vs transfer plot: no trials at or after transfer_start={transfer_start}")


def print_summary_statistics(participant_df: pd.DataFrame,
                            learning_df: pd.DataFrame,
                            config: Dict,
                            experiment_type: str = 'single',
                            trial_df: pd.DataFrame = None,
                            stars: List[str] = None,
                            start_trial: int = 0,
                            end_trial: int = None) -> None:
    """
    Print summary statistics to console.

    Args:
        participant_df: Participant summary data
        learning_df: Learning curve data
        config: Experiment configuration
    """
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    def _fmt_int(value: Any) -> Any:
        """Format numpy numeric types without trailing decimals."""
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    # Experiment info
    model_cfg = config.get('model', {})
    experiment_cfg = config.get('experiment', {})
    model_family = model_cfg.get('family', 'UnknownModel')
    model_size = model_cfg.get('size', 'UnknownSize')
    goal_star_label = _format_goal_star_label(config, experiment_type, stars)

    print(f"\nModel: {model_family}-{model_size}")
    print(f"Goal Star: {goal_star_label}")
    print(f"Number of Participants: {len(participant_df)}")
    configured_trials = experiment_cfg.get('n_trials', 'Unknown')
    if start_trial > 0 or end_trial is not None:
        if start_trial > 0 and end_trial is not None:
            print(f"Trial Range: {start_trial} to {end_trial}")
        elif start_trial > 0:
            print(f"Trial Range: {start_trial}+ (starting from trial {start_trial})")
        else:
            print(f"Trial Range: 0 to {end_trial}")
        print(f"Configured Trials per Participant: {configured_trials}")
    else:
        print(f"Trials per Participant: {configured_trials}")

    # Overall statistics
    print("\n--- Overall Performance ---")
    total_trials = participant_df['total_trials'].sum()
    total_successes = participant_df['successful_trials'].sum()
    overall_success_rate = total_successes / total_trials
    print(f"Total Trials: {total_trials}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {overall_success_rate:.3f} ({overall_success_rate*100:.1f}%)")

    # Participant statistics
    print("\n--- Participant Statistics ---")
    print(f"Mean Success Rate: {participant_df['success_rate'].mean():.3f}")
    print(f"Median Success Rate: {participant_df['success_rate'].median():.3f}")
    print(f"Std Dev: {participant_df['success_rate'].std():.3f}")
    min_participant = _fmt_int(participant_df.loc[participant_df['success_rate'].idxmin(), 'participant_id'])
    max_participant = _fmt_int(participant_df.loc[participant_df['success_rate'].idxmax(), 'participant_id'])
    print(f"Min Success Rate: {participant_df['success_rate'].min():.3f} (Participant {min_participant})")
    print(f"Max Success Rate: {participant_df['success_rate'].max():.3f} (Participant {max_participant})")

    # Learning trend
    print("\n--- Learning Trend ---")
    first_5_trials = learning_df.head(5)['cumulative_success_rate'].mean()
    last_5_trials = learning_df.tail(5)['cumulative_success_rate'].mean()
    print(f"Avg Success Rate (First 5 trials): {first_5_trials:.3f}")
    print(f"Avg Success Rate (Last 5 trials): {last_5_trials:.3f}")
    improvement = last_5_trials - first_5_trials
    print(f"Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")

    # Top performers
    print("\n--- Top 3 Performers ---")
    top_3 = participant_df.nlargest(3, 'success_rate')
    for idx, row in top_3.iterrows():
        participant_label = _fmt_int(row['participant_id'])
        success_count = _fmt_int(row['successful_trials'])
        total_count = _fmt_int(row['total_trials'])
        print(f"  Participant {participant_label}: {row['success_rate']:.3f} ({success_count}/{total_count} trials)")

    print("\n" + "="*70 + "\n")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze old format inference batch JSON results from star-making game experiments.'
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to the inference batch JSON file (old format with conversation_history)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='analysis_output',
        help='Output directory for analysis results (default: analysis_output)'
    )
    parser.add_argument(
        '--no-lines',
        action='store_true',
        help='Show only markers without connecting lines in learning curves'
    )
    parser.add_argument(
        '--start-trial',
        type=int,
        default=0,
        help='Start analysis from this trial number (default: 0, analyze all trials)'
    )
    parser.add_argument(
        '--end-trial',
        type=int,
        default=None,
        help='End analysis at this trial number (inclusive, default: None, analyze all trials)'
    )
    parser.add_argument(
        '--transfer-start',
        type=int,
        default=80,
        help='Trial index where transfer phase starts (default: 80)'
    )

    args = parser.parse_args()

    # Validate input file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")

    # Load data
    data = load_json_data(args.json_file)

    # Extract data
    print("\nExtracting data from conversation history...")
    participant_df = extract_participant_data(data)
    trial_df = extract_trial_details(data)

    # Filter trials based on start_trial and end_trial arguments
    if args.start_trial > 0 or args.end_trial is not None:
        original_trial_count = len(trial_df)

        # Apply start trial filter
        if args.start_trial > 0:
            trial_df = trial_df[trial_df['trial'] >= args.start_trial].copy()

        # Apply end trial filter
        if args.end_trial is not None:
            trial_df = trial_df[trial_df['trial'] <= args.end_trial].copy()

        filtered_trial_count = len(trial_df)

        # Format the range message
        if args.start_trial > 0 and args.end_trial is not None:
            range_msg = f"trials {args.start_trial} to {args.end_trial}"
        elif args.start_trial > 0:
            range_msg = f"trial {args.start_trial} onwards"
        else:
            range_msg = f"trials up to {args.end_trial}"

        print(f"\n✓ Filtered trials: analyzing {range_msg}")
        print(f"  (Excluded {original_trial_count - filtered_trial_count} trials, analyzing {filtered_trial_count} trials)")

        # Recalculate participant-level statistics based on filtered trials
        participant_summary = []
        for participant_id in trial_df['participant_id'].unique():
            p_trials = trial_df[trial_df['participant_id'] == participant_id]
            total_trials = len(p_trials)
            successful_trials = p_trials['success'].sum()
            success_rate = successful_trials / total_trials if total_trials > 0 else 0.0

            participant_summary.append({
                'participant_id': participant_id,
                'total_trials': total_trials,
                'successful_trials': int(successful_trials),
                'success_rate': success_rate
            })
        participant_df = pd.DataFrame(participant_summary)

    learning_df = calculate_learning_curves(trial_df)
    success_matrix = create_success_matrix(trial_df)
    print("✓ Data extraction complete")

    # Detect experiment type
    experiment_type, stars = detect_experiment_type(trial_df)
    print(f"\nExperiment type: {experiment_type.upper()}")
    if experiment_type == 'multi':
        print(f"Stars detected: {', '.join(stars)}")

    # Save CSV outputs
    print("\nSaving CSV files...")
    save_csv_outputs(participant_df, learning_df, trial_df, output_dir,
                    experiment_type, stars)

    # Create visualizations
    print("\nGenerating visualizations...")
    show_lines = not args.no_lines
    create_visualizations(participant_df, learning_df, success_matrix, trial_df,
                         data['config'], output_dir, experiment_type, stars, show_lines,
                         args.start_trial, args.end_trial, args.transfer_start)

    # Print summary statistics
    print_summary_statistics(participant_df, learning_df, data['config'],
                            experiment_type, trial_df, stars, args.start_trial, args.end_trial)

    print(f"✓ Analysis complete! Results saved to {output_dir.absolute()}")
    return 0


if __name__ == '__main__':
    exit(main())


"""
Sample usage:
    # Analyze a single experiment file
    python analysis/analyze_result.py \
        results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json \
        -o results/gpt-oss-120b/metrics_plots/transfer_high/item

    # Analyze all 4 conditions
    python analysis/analyze_result.py results/gpt-oss-120b/transfer_low/no_item_20251122_040337_experiment.json -o results/gpt-oss-120b/metrics_plots/transfer_low/no_item
    python analysis/analyze_result.py results/gpt-oss-120b/transfer_low/item_20251122_040528_experiment.json -o results/gpt-oss-120b/metrics_plots/transfer_low/item
    python analysis/analyze_result.py results/gpt-oss-120b/transfer_high/no_item_20251122_075552_experiment.json -o results/gpt-oss-120b/metrics_plots/transfer_high/no_item
    python analysis/analyze_result.py results/gpt-oss-120b/transfer_high/item_20251122_080413_experiment.json -o results/gpt-oss-120b/metrics_plots/transfer_high/item

    # With options
    python analysis/analyze_result.py path/to/experiment.json -o output_dir --no-lines
    python analysis/analyze_result.py path/to/experiment.json -o output_dir --start-trial 80
    python analysis/analyze_result.py path/to/experiment.json -o output_dir --start-trial 20 --end-trial 80
    python analysis/analyze_result.py path/to/experiment.json -o output_dir --transfer-start 100
"""
