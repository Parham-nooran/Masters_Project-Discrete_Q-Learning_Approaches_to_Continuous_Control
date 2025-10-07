"""
Standalone script to plot DecQN training results from saved metrics.
Run this after training to generate plots from saved metrics files.

Usage:
    python plot_results.py
    python plot_results.py --metrics_file custom_metrics.json
    python plot_results.py --window 50  # Change smoothing window
"""

import argparse

from src.plotting.plotting_utils import PlottingUtils
from src.common.metrics_tracker import MetricsTracker


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    parser = argparse.ArgumentParser(description="Plot DecQN training results")
    parser.add_argument(
        "--metrics_file",
        default="metrics/metrics.pkl",
        help="Path to metrics pkl file",
    )
    parser.add_argument(
        "--window", type=int, default=100, help="Window size for running average"
    )
    parser.add_argument("--output_dir", default="plots", help="Directory to save plots")

    args = parser.parse_args()

    print("DecQN Results Plotting Tool")
    print("=" * 30)
    print(f"Loading metrics from: {args.metrics_file}")
    print(f"Smoothing window: {args.window}")
    print(f"Output directory: {args.output_dir}")

    tracker = MetricsTracker()
    try:
        tracker.load_metrics(args.metrics_file)

        if not tracker.episode_rewards:
            print(f"❌ No metrics found in {args.metrics_file}")
            print("Make sure you have run training and saved metrics first.")
            return

        print(f"✅ Loaded {len(tracker.episode_rewards)} episodes of data")

        plotter = PlottingUtils(tracker, save_dir=args.output_dir)

        print("\nGenerating plots...")
        plotter.plot_training_curves(window=args.window, save=True)
        plotter.plot_loss_comparison(window=args.window, save=True)
        plotter.plot_td_error_analysis(window=args.window, save=True)
        plotter.plot_reward_distribution(save=True)

        print("\nTraining Summary:")
        plotter.print_summary_stats()

        print(f"\nAll plots saved to {args.output_dir}/")

    except Exception as e:
        print(f"❌ Error loading metrics: {e}")


if __name__ == "__main__":
    main()
