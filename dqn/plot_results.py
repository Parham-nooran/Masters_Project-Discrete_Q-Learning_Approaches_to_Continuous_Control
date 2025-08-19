"""
Standalone script to plot DecQN training results from saved metrics.
Run this after training to generate plots from saved metrics files.

Usage:
    python plot_results.py
    python plot_results.py --metrics_file custom_metrics.json
    python plot_results.py --window 50  # Change smoothing window
"""

import argparse

from plotting_utils import MetricsTracker, PlottingUtils


def main():
    parser = argparse.ArgumentParser(description='Plot DecQN training results')
    parser.add_argument('--metrics_file',
                        default='metrics/training_metrics.json',
                        help='Path to metrics JSON file')
    parser.add_argument('--window',
                        type=int,
                        default=100,
                        help='Window size for running average')
    parser.add_argument('--output_dir',
                        default='plots',
                        help='Directory to save plots')

    args = parser.parse_args()

    print("DecQN Results Plotting Tool")
    print("=" * 30)
    print(f"Loading metrics from: {args.metrics_file}")
    print(f"Smoothing window: {args.window}")
    print(f"Output directory: {args.output_dir}")

    # Load and plot
    tracker = MetricsTracker()
    try:
        tracker.load_metrics(args.metrics_file)

        if not tracker.episode_rewards:
            print(f"❌ No metrics found in {args.metrics_file}")
            print("Make sure you have run training and saved metrics first.")
            return

        print(f"✅ Loaded {len(tracker.episode_rewards)} episodes of data")

        # Create plotter and generate all plots
        plotter = PlottingUtils(tracker, save_dir=args.output_dir)

        print("\n📊 Generating plots...")
        plotter.plot_training_curves(window=args.window, save=True)
        plotter.plot_reward_distribution(save=True)

        print("\n📈 Training Summary:")
        plotter.print_summary_stats()

        print(f"\n✅ All plots saved to {args.output_dir}/")

    except FileNotFoundError:
        print(f"❌ Metrics file not found: {args.metrics_file}")
        print("Make sure you have run training first to generate metrics.")
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")


if __name__ == "__main__":
    main()
