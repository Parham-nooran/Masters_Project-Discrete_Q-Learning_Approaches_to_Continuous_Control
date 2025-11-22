"""
Enhanced plotting script for comparing multiple RL algorithms.
Focuses on reward plots with elegant, publication-quality visualizations.
Groups metrics by task and plots all algorithms for each task.

Usage:
    python plot_comparison.py --metrics_dir metrics/
    python plot_comparison.py --metrics_dir metrics/ --window 50
    python plot_comparison.py --metrics_dir metrics/ --tasks walker_walk cheetah_run
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

COLORS = [
    '#2E86AB',
    '#A23B72',
    '#F18F01',
    '#C73E1D',
    '#6A994E',
    '#BC4B51',
    '#8B5A3C',
    '#4A5859',
    '#5E548E',
    '#E07A5F',
]


class MetricsLoader:
    """Load and parse metrics files."""

    @staticmethod
    def parse_filename(filename: str) -> Tuple[str, str]:
        """
        Parse filename to extract algorithm and task.
        Expected formats:
            - algorithm_task1_task2.pkl (e.g., gqn_walker_walk.pkl)
            - algorithm1_algorithm2_task1_task2.pkl (e.g., bangbang_mpo_walker_walk.pkl)

        Returns:
            Tuple of (algorithm, task)

        Raises:
            ValueError: If filename format is invalid
        """
        if not filename.endswith('.pkl'):
            raise ValueError(f"File must be .pkl format: {filename}")

        name = filename[:-4]
        parts = name.split('_')

        if len(parts) < 3:
            raise ValueError(
                f"Invalid filename format: {filename}. "
                f"Expected format: algorithm_taskpart1_taskpart2.pkl or "
                f"algorithm1_algorithm2_taskpart1_taskpart2.pkl"
            )

        """
        Strategy: Look for common task patterns to determine split point.
        Common task domains: walker, cheetah, hopper, humanoid, reacher, finger, etc.
        If we find these in the parts, everything before is algorithm, everything from that point is task.
        """
        task_domains = {
            'walker', 'cheetah', 'hopper', 'humanoid', 'reacher',
            'finger', 'cartpole', 'acrobot', 'pendulum', 'swimmer',
            'ant', 'halfcheetah', 'standup', 'pointmass'
        }

        split_idx = None
        for i, part in enumerate(parts):
            if part.lower() in task_domains:
                split_idx = i
                break

        if split_idx is None:
            """
            If no known domain found, assume last two parts are task (e.g., part1_part2)
            and everything before is algorithm
            """
            if len(parts) >= 3:
                split_idx = len(parts) - 2
            else:
                split_idx = 1

        algorithm_parts = parts[:split_idx]
        task_parts = parts[split_idx:]

        if not algorithm_parts or not task_parts:
            raise ValueError(
                f"Could not properly parse algorithm and task from: {filename}"
            )

        algorithm = '_'.join(algorithm_parts).upper()
        task = '_'.join(task_parts)

        return algorithm, task

    @staticmethod
    def load_metrics(filepath: str) -> Dict:
        """Load metrics from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def compute_total_steps(episode_steps: List[int]) -> np.ndarray:
        """Convert episode steps to cumulative total steps."""
        return np.cumsum(episode_steps)


class RewardPlotter:
    """Create publication-quality reward plots."""

    def __init__(self, output_dir: str = './output/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_task_comparison(
            self,
            algorithms_data: Dict[str, Dict],
            task: str,
            window: int = 100
    ):
        """
        Plot comparison of multiple algorithms for a specific task.

        Args:
            algorithms_data: Dictionary mapping algorithm names to their data
            task: Name of the task being plotted
            window: Window size for moving average smoothing
        """
        num_algorithms = len(algorithms_data)

        if num_algorithms == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

        sorted_algorithms = sorted(algorithms_data.keys())

        for idx, algorithm in enumerate(sorted_algorithms):
            color = COLORS[idx % len(COLORS)]
            data = algorithms_data[algorithm]
            steps = data['steps'] / 1e6
            rewards = data['rewards']

            axes[0].plot(
                steps, rewards,
                alpha=0.4,
                color=color,
                linewidth=0.8,
                label=f'{algorithm}'
            )

            if len(rewards) > window:
                smoothed = self._compute_moving_average(rewards, window)
                std = self._compute_moving_std(rewards, window)
                steps_smoothed = steps[window - 1:]

                axes[1].plot(
                    steps_smoothed, smoothed,
                    color=color,
                    linewidth=2.5,
                    label=f'{algorithm}',
                    alpha=0.9
                )
                axes[1].fill_between(
                    steps_smoothed,
                    smoothed - std,
                    smoothed + std,
                    alpha=0.15,
                    color=color
                )
            else:
                axes[1].plot(
                    steps, rewards,
                    color=color,
                    linewidth=2.5,
                    label=f'{algorithm}',
                    alpha=0.9
                )

        task_title = task.replace("_", " ").title()

        axes[0].set_xlabel('Training Steps (Millions)', fontweight='bold')
        axes[0].set_ylabel('Episode Reward', fontweight='bold')
        axes[0].set_title(f'{task_title} - Raw Episode Rewards', fontweight='bold', pad=15)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='best', framealpha=0.95, edgecolor='gray', fancybox=True)

        axes[1].set_xlabel('Training Steps (Millions)', fontweight='bold')
        axes[1].set_ylabel('Episode Reward', fontweight='bold')
        axes[1].set_title(
            f'{task_title} - Smoothed Rewards (MA={window})',
            fontweight='bold',
            pad=15
        )
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='best', framealpha=0.95, edgecolor='gray', fancybox=True)

        for ax in axes:
            ax.set_facecolor('#FAFAFA')

        plt.tight_layout()

        filename = f'{task}_algorithm_comparison.pdf'
        filepath = self.output_dir / filename
        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"  Saved: {filename}")

    @staticmethod
    def _compute_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving average."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')

    @staticmethod
    def _compute_moving_std(data: np.ndarray, window: int) -> np.ndarray:
        """Compute moving standard deviation."""
        if len(data) < window:
            return np.zeros_like(data)

        result = []
        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1:i + 1]
            result.append(np.std(window_data))
        return np.array(result)


def main():
    parser = argparse.ArgumentParser(
        description='Plot and compare RL algorithm results grouped by task'
    )
    parser.add_argument(
        '--metrics_dir',
        type=str,
        default='metrics/',
        help='Directory containing metrics .pkl files'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=100,
        help='Window size for moving average'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='*',
        default=None,
        help='Filter by specific tasks (optional, space-separated)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/plots',
        help='Directory to save plots'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RL Algorithm Comparison Plotter - Grouped by Task")
    print("=" * 70)
    print(f"Metrics directory: {args.metrics_dir}")
    print(f"Smoothing window: {args.window}")
    print(f"Output directory: {args.output_dir}")
    if args.tasks:
        print(f"Filtering tasks: {', '.join(args.tasks)}")
    print("=" * 70)

    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.exists():
        print(f"\nError: Metrics directory not found: {metrics_dir}")
        return

    pkl_files = list(metrics_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"\nError: No .pkl files found in {metrics_dir}")
        return

    print(f"\nFound {len(pkl_files)} metrics file(s)")

    loader = MetricsLoader()
    plotter = RewardPlotter(args.output_dir)

    task_algorithms = defaultdict(dict)
    skipped_files = []

    print("\n" + "-" * 70)
    print("Processing metrics files...")
    print("-" * 70)

    for pkl_file in pkl_files:
        try:
            algorithm, task = loader.parse_filename(pkl_file.name)

            if args.tasks and task not in args.tasks:
                continue

            print(f"\n{pkl_file.name}")
            print(f"  Algorithm: {algorithm}")
            print(f"  Task: {task}")

            metrics = loader.load_metrics(str(pkl_file))

            if 'episode_rewards' not in metrics or 'episode_steps' not in metrics:
                print(f"  Warning: Missing required fields, skipping...")
                skipped_files.append((pkl_file.name, "Missing required fields"))
                continue

            episode_rewards = np.array(metrics['episode_rewards'])
            episode_steps = np.array(metrics['episode_steps'])

            if len(episode_rewards) == 0 or len(episode_steps) == 0:
                print(f"  Warning: Empty data, skipping...")
                skipped_files.append((pkl_file.name, "Empty data"))
                continue

            total_steps = loader.compute_total_steps(episode_steps)

            print(f"  Episodes: {len(episode_rewards):,}")
            print(f"  Total steps: {total_steps[-1]:,}")
            print(f"  Mean reward: {episode_rewards.mean():.2f} +/- {episode_rewards.std():.2f}")
            print(f"  Max reward: {episode_rewards.max():.2f}")
            print(f"  Min reward: {episode_rewards.min():.2f}")

            task_algorithms[task][algorithm] = {
                'steps': total_steps,
                'rewards': episode_rewards,
                'filename': pkl_file.name
            }

        except ValueError as e:
            print(f"\nSkipping {pkl_file.name}")
            print(f"  Reason: {e}")
            skipped_files.append((pkl_file.name, str(e)))
            continue
        except Exception as e:
            print(f"\nError processing {pkl_file.name}")
            print(f"  Error: {e}")
            skipped_files.append((pkl_file.name, f"Error: {e}"))
            continue

    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    if not task_algorithms:
        print("\nNo valid metrics found to plot!")
        if skipped_files:
            print("\nSkipped files:")
            for filename, reason in skipped_files:
                print(f"  {filename}: {reason}")
        return

    for task_idx, (task, algorithms_data) in enumerate(sorted(task_algorithms.items()), 1):
        print(f"\n[{task_idx}/{len(task_algorithms)}] Task: {task}")
        print(f"  Algorithms: {', '.join(sorted(algorithms_data.keys()))}")
        print(f"  Generating comparison plot...")

        plotter.plot_task_comparison(
            algorithms_data,
            task,
            window=args.window
        )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Tasks plotted: {len(task_algorithms)}")
    print(f"Total algorithms: {sum(len(algs) for algs in task_algorithms.values())}")
    print(f"Output directory: {args.output_dir}")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} file(s):")
        for filename, reason in skipped_files:
            print(f"  {filename}")
            print(f"    {reason}")

    print("\n" + "=" * 70)
    print("All plots generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
