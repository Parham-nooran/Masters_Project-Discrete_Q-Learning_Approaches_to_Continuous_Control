"""
Standalone script to plot DecQN training results from saved metrics.
Run this after training to generate plots from saved metrics files.

Usage:
    python plot_results.py
    python plot_results.py --metrics_file custom_metrics.json
    python plot_results.py --window 50  # Change smoothing window
"""

import argparse

from src.common.logger import Logger
from src.plotting.plotting_utils import PlottingUtils
from src.common.metrics_tracker import MetricsTracker


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot DecQN training results")
    parser.add_argument(
        "--metrics_file",
        default="metrics/metrics.pkl",
        help="Path to metrics pkl file",
    )
    parser.add_argument(
        "--window", type=int, default=100, help="Window size for running average"
    )
    parser.add_argument(
        "--output_dir", default="./output/plots", help="Directory to save plots"
    )

    return parser.parse_args()


class Plotter(Logger):
    def __init__(self, working_dir="./src/plotting/output"):
        super().__init__(working_dir + "/logs")
        self.tracker = MetricsTracker(self.logger, save_dir=self.working_dir)
        self.args = parse_arguments()
        self.log_important_args()

    def log_important_args(self):
        self.logger.info("DecQN Results Plotting Tool")
        self.logger.info("=" * 30)
        self.logger.info(f"Loading metrics from: {self.args.metrics_file}")
        self.logger.info(f"Smoothing window: {self.args.window}")
        self.logger.info(f"Output directory: {self.args.output_dir}")

    def plot(self):
        try:
            self.tracker.load_metrics(self.args.metrics_file)
        except Exception as e:
            self.logger.warning(f"Error loading metrics: {e}")
        if not self.tracker.episode_rewards:
            self.logger.info(f"No metrics found in {self.args.metrics_file}")
            self.logger.info("Make sure you have run training and saved metrics first.")
            return

        self.logger.info(f"Loaded {len(self.tracker.episode_rewards)} episodes of data")

        plotter = PlottingUtils(self.logger, self.tracker, self.args.output_dir)

        self.logger.info("Generating plots...")
        plotter.plot_training_curves(window=self.args.window, save=True)
        plotter.plot_loss_comparison(window=self.args.window, save=True)
        plotter.plot_td_error_analysis(window=self.args.window, save=True)
        plotter.plot_reward_distribution(save=True)

        self.logger.info("Training Summary:")
        plotter.print_summary_stats()

        self.logger.info(f"All plots saved to {self.args.output_dir}/")


if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot()
