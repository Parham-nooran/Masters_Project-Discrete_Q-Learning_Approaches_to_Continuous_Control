"""
Enhanced plotting script for comparing multiple RL algorithms with seed averaging.
Groups metrics by task and algorithm, averages over seeds, and plots comparisons.

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
from scipy import interpolate

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
    def parse_filename(filename: str) -> Tuple[str, str, int]:
        """
        Parse filename to extract algorithm, task, and seed.
        Expected formats:
            - algorithm_task1_task2_seed.pkl (e.g., gqn_walker_walk_0.pkl)
            - algorithm1_algorithm2_task1_task2_seed.pkl (e.g., bangbang_mpo_walker_walk_42.pkl)

        Returns:
            Tuple of (algorithm, task, seed)

        Raises:
            ValueError: If filename format is invalid
        """
        if not filename.endswith('.pkl'):
            raise ValueError(f"File must be .pkl format: {filename}")

        name = filename[:-4]
        parts = name.split('_')

        if len(parts) < 4:
            raise ValueError(
                f"Invalid filename format: {filename}. "
                f"Expected format: algorithm_taskpart1_taskpart2_seed.pkl"
            )

        try:
            seed = int(parts[-1])
        except ValueError:
            raise ValueError(
                f"Last part must be seed number: {filename}"
            )

        parts_without_seed = parts[:-1]

        task_domains = {
            'walker', 'cheetah', 'hopper', 'humanoid', 'reacher',
            'finger', 'cartpole', 'acrobot', 'pendulum', 'swimmer',
            'ant', 'halfcheetah', 'standup', 'pointmass', 'fish'
        }

        split_idx = None
        for i, part in enumerate(parts_without_seed):
            if part.lower() in task_domains:
                split_idx = i
                break

        if split_idx is None:
            if len(parts_without_seed) >= 3:
                split_idx = len(parts_without_seed) - 2
            else:
                split_idx = 1

        algorithm_parts = parts_without_seed[:split_idx]
        task_parts = parts_without_seed[split_idx:]

        if not algorithm_parts or not task_parts:
            raise ValueError(
                f"Could not properly parse algorithm and task from: {filename}"
            )

        algorithm = '_'.join(algorithm_parts).upper()
        task = '_'.join(task_parts)

        return algorithm, task, seed

    @staticmethod
    def load_metrics(filepath: str) -> Dict:
        """Load metrics from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def compute_total_steps(episode_steps: List[int]) -> np.ndarray:
        """Convert episode steps to cumulative total steps."""
        return np.cumsum(episode_steps)


class SeedAverager:
    """Average metrics across multiple seeds."""

    @staticmethod
    def interpolate_to_common_steps(seeds_data: List[Dict], num_points: int = 1000) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate all seeds to common step points for averaging.

        Args:
            seeds_data: List of dictionaries containing 'steps' and 'rewards'
            num_points: Number of interpolation points

        Returns:
            Tuple of (common_steps, mean_rewards, std_rewards)
        """
        min_steps = min(min(data['steps'][-1] for data in seeds_data), 1e6)

        common_steps = np.linspace(0, min_steps, num_points)

        interpolated_rewards = []
        for data in seeds_data:
            f = interpolate.interp1d(
                data['steps'],
                data['rewards'],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            interpolated_rewards.append(f(common_steps))

        interpolated_rewards = np.array(interpolated_rewards)
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)

        return common_steps, mean_rewards, std_rewards


class RewardPlotter:
    """Create publication-quality reward plots."""

    def __init__(self, output_dir: str = './output/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        self.averager = SeedAverager()

    def plot_seed_averaged_comparison(
            self,
            algorithms_data: Dict[str, List[Dict]],
            task: str
    ):
        """
        Plot 1: Seed-averaged comparison (no window smoothing).
        Each algorithm shows mean and std over seeds.

        Args:
            algorithms_data: Dict mapping algorithm names to list of seed data
            task: Name of the task being plotted
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        sorted_algorithms = sorted(algorithms_data.keys())

        for idx, algorithm in enumerate(sorted_algorithms):
            color = COLORS[idx % len(COLORS)]
            seeds_data = algorithms_data[algorithm]

            if len(seeds_data) == 0:
                continue

            common_steps, mean_rewards, std_rewards = self.averager.interpolate_to_common_steps(seeds_data)
            steps_m = common_steps / 1e6
            ax.plot(
                steps_m, mean_rewards,
                color=color,
                linewidth=2.5,
                label=f'{algorithm} (n={len(seeds_data)})',
                alpha=0.9
            )
            ax.fill_between(
                steps_m,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2,
                color=color
            )

        task_title = task.replace("_", " ").title()

        ax.set_xlabel('Training Steps (Millions)', fontweight='bold')
        ax.set_ylabel('Episode Reward', fontweight='bold')
        ax.set_title(
            f'{task_title} - Seed-Averaged Rewards',
            fontweight='bold',
            pad=15
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fancybox=True)
        ax.set_facecolor('#FAFAFA')

        plt.tight_layout()

        filename = f'{task}_seed_averaged.pdf'
        filepath = self.output_dir / filename
        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"  Saved: {filename}")

    def plot_smoothed_seed_averaged_comparison(
            self,
            algorithms_data: Dict[str, List[Dict]],
            task: str,
            window: int = 100
    ):
        """
        Plot 2: Seed-averaged + window-smoothed comparison.
        First averages over seeds, then applies moving average smoothing.

        Args:
            algorithms_data: Dict mapping algorithm names to list of seed data
            task: Name of the task being plotted
            window: Window size for moving average
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        sorted_algorithms = sorted(algorithms_data.keys())

        for idx, algorithm in enumerate(sorted_algorithms):
            color = COLORS[idx % len(COLORS)]
            seeds_data = algorithms_data[algorithm]

            if len(seeds_data) == 0:
                continue

            common_steps, mean_rewards, std_rewards = self.averager.interpolate_to_common_steps(seeds_data)

            if len(mean_rewards) > window:
                smoothed_mean = self._compute_moving_average(mean_rewards, window)
                smoothed_std = self._compute_moving_average(std_rewards, window)
                steps_smoothed = common_steps[window - 1:] / 1e6
                ax.plot(
                    steps_smoothed, smoothed_mean,
                    color=color,
                    linewidth=2.5,
                    label=f'{algorithm} (n={len(seeds_data)})',
                    alpha=0.9
                )
                ax.fill_between(
                    steps_smoothed,
                    smoothed_mean - smoothed_std,
                    smoothed_mean + smoothed_std,
                    alpha=0.2,
                    color=color
                )
            else:
                steps_m = common_steps / 1e6
                ax.plot(
                    steps_m, mean_rewards,
                    color=color,
                    linewidth=2.5,
                    label=f'{algorithm} (n={len(seeds_data)})',
                    alpha=0.9
                )
                ax.fill_between(
                    steps_m,
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    alpha=0.2,
                    color=color
                )

        task_title = task.replace("_", " ").title()

        ax.set_xlabel('Training Steps (Millions)', fontweight='bold')
        ax.set_ylabel('Episode Reward', fontweight='bold')
        ax.set_title(
            f'{task_title} - Smoothed Seed-Averaged Rewards (MA={window})',
            fontweight='bold',
            pad=15
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fancybox=True)
        ax.set_facecolor('#FAFAFA')

        plt.tight_layout()

        filename = f'{task}_smoothed_seed_averaged.pdf'
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


def main():
    parser = argparse.ArgumentParser(
        description='Plot and compare RL algorithm results with seed averaging'
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
        help='Window size for moving average smoothing'
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
    print("RL Algorithm Comparison Plotter - Seed Averaging")
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

    task_algorithm_seeds = defaultdict(lambda: defaultdict(list))
    skipped_files = []

    print("\n" + "-" * 70)
    print("Processing metrics files...")
    print("-" * 70)

    for pkl_file in pkl_files:
        try:
            algorithm, task, seed = loader.parse_filename(pkl_file.name)

            if args.tasks and task not in args.tasks:
                continue

            print(f"\n{pkl_file.name}")
            print(f"  Algorithm: {algorithm}")
            print(f"  Task: {task}")
            print(f"  Seed: {seed}")

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
            task_algorithm_seeds[task][algorithm].append({
                'steps': total_steps,
                'rewards': episode_rewards,
                'seed': seed,
                'filename': pkl_file.name
            })

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

    if not task_algorithm_seeds:
        print("\nNo valid metrics found to plot!")
        if skipped_files:
            print("\nSkipped files:")
            for filename, reason in skipped_files:
                print(f"  {filename}: {reason}")
        return

    for task_idx, (task, algorithms_data) in enumerate(sorted(task_algorithm_seeds.items()), 1):
        print(f"\n[{task_idx}/{len(task_algorithm_seeds)}] Task: {task}")

        for algo, seeds in algorithms_data.items():
            print(f"  {algo}: {len(seeds)} seed(s)")

        print(f"\n  Generating seed-averaged plot...")
        plotter.plot_seed_averaged_comparison(algorithms_data, task)

        print(f"  Generating smoothed seed-averaged plot...")
        plotter.plot_smoothed_seed_averaged_comparison(algorithms_data, task, window=args.window)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Tasks plotted: {len(task_algorithm_seeds)}")

    total_algorithms = sum(len(algs) for algs in task_algorithm_seeds.values())
    total_runs = sum(len(seeds) for algs in task_algorithm_seeds.values() for seeds in algs.values())

    print(f"Total unique algorithm-task pairs: {total_algorithms}")
    print(f"Total runs (including seeds): {total_runs}")
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
