"""Utilities for plotting training results."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _filter_valid_data(episodes, data):
    """Filter out None values from data."""
    return [(ep, val) for ep, val in zip(episodes, data) if val is not None]


def _save_figure(save_dir, filename):
    """Save figure to file."""
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")


class PlottingUtils:
    """Utility class for plotting training metrics."""

    def __init__(self, logger, metrics, save_dir):
        self.logger = logger
        self.metrics = metrics
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self, window=100, save=False, save_dir=None):
        """Plot comprehensive training curves."""
        save_dir = save_dir or self.save_dir
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))

        self._plot_raw_rewards(axes[0, 0])
        # self._plot_moving_average_rewards(axes[0, 1], window)
        self._plot_training_loss(axes[1, 0])
        self._plot_mse_loss(axes[1, 1])
        self._plot_mean_abs_td_error(axes[2, 0])
        self._plot_mean_squared_td_error(axes[2, 1])
        self._plot_q_value_means(axes[3, 0])
        self._plot_epsilon_decay(axes[3, 1])

        plt.tight_layout()
        if save:
            _save_figure(save_dir, "training_curves.png")

    def _plot_raw_rewards(self, ax):
        """Plot raw episode rewards."""
        rewards = []
        for episode_reward, episode_step in zip(self.metrics.episode_rewards, self.metrics.episode_steps):
            rewards += ([episode_reward] * episode_step)
        ax.plot(rewards)
        ax.set_title("Episode Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)

    def _plot_moving_average_rewards(self, ax, window):
        """Plot moving average of rewards."""
        if len(self.metrics.episode_rewards) <= 10:
            return

        window_size = min(window, len(self.metrics.episode_rewards) // 10)
        moving_avg = np.convolve(
            self.metrics.episode_rewards,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        ax.plot(self.metrics.episodes[window_size - 1:], moving_avg)
        ax.set_title(f"Moving Average Rewards (window={window_size})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.grid(True)

    def _plot_training_loss(self, ax):
        """Plot training loss."""
        valid_losses = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_losses
        )
        if valid_losses:
            episodes, losses = zip(*valid_losses)
            ax.plot(episodes, losses)
            ax.set_title("Training Loss (Huber)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Loss")
            ax.grid(True)

    def _plot_mse_loss(self, ax):
        """Plot MSE loss."""
        valid_mse = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_mse_losses
        )
        if valid_mse:
            episodes, mse_losses = zip(*valid_mse)
            ax.plot(episodes, mse_losses)
            ax.set_title("MSE Loss")
            ax.set_xlabel("Episode")
            ax.set_ylabel("MSE")
            ax.grid(True)

    def _plot_mean_abs_td_error(self, ax):
        """Plot mean absolute TD error."""
        valid_td_abs = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_mean_abs_td_error
        )
        if valid_td_abs:
            episodes, td_errors = zip(*valid_td_abs)
            ax.plot(episodes, td_errors, color="orange")
            ax.set_title("Mean Absolute TD Error")
            ax.set_xlabel("Episode")
            ax.set_ylabel("TD Error")
            ax.grid(True)

    def _plot_mean_squared_td_error(self, ax):
        """Plot mean squared TD error."""
        valid_td_sq = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_mean_squared_td_error
        )
        if valid_td_sq:
            episodes, td_squared = zip(*valid_td_sq)
            ax.plot(episodes, td_squared, color="red")
            ax.set_title("Mean Squared TD Error")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Squared TD Error")
            ax.grid(True)

    def _plot_q_value_means(self, ax):
        """Plot Q-value means."""
        valid_q_means = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_q_means
        )
        if valid_q_means:
            episodes, q_means = zip(*valid_q_means)
            ax.plot(episodes, q_means, color="green")
            ax.set_title("Q-value Means")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Q-mean")
            ax.grid(True)

    def _plot_epsilon_decay(self, ax):
        """Plot epsilon decay."""
        ax.plot(self.metrics.episodes, self.metrics.episode_epsilons, color="purple")
        ax.set_title("Epsilon Decay")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.grid(True)

    def plot_loss_comparison(self, window=50, save=False, save_dir=None):
        """Plot Huber loss vs MSE loss comparison."""
        save_dir = save_dir or self.save_dir
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        self._plot_huber_loss_comparison(axes[0, :], window)
        self._plot_mse_loss_comparison(axes[1, :], window)

        plt.tight_layout()
        if save:
            _save_figure(save_dir, "loss_comparison.png")

    def _plot_huber_loss_comparison(self, axes, window):
        """Plot Huber loss raw and smoothed."""
        valid_losses = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_losses
        )
        if not valid_losses:
            return

        episodes, losses = zip(*valid_losses)
        axes[0].plot(episodes, losses)
        axes[0].set_title("Huber Loss (Raw)")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            axes[1].plot(episodes[window - 1:], smoothed)
            axes[1].set_title(f"Huber Loss (Smoothed, window={window})")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True)

    def _plot_mse_loss_comparison(self, axes, window):
        """Plot MSE loss raw and smoothed."""
        valid_mse = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_mse_losses
        )
        if not valid_mse:
            return

        episodes, mse_losses = zip(*valid_mse)
        axes[0].plot(episodes, mse_losses, color="red")
        axes[0].set_title("MSE Loss (Raw)")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("MSE")
        axes[0].grid(True)

        if len(mse_losses) > window:
            smoothed = np.convolve(mse_losses, np.ones(window) / window, mode="valid")
            axes[1].plot(episodes[window - 1:], smoothed, color="red")
            axes[1].set_title(f"MSE Loss (Smoothed, window={window})")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("MSE")
            axes[1].grid(True)

    def plot_td_error_analysis(self, window=50, save=False, save_dir=None):
        """Plot detailed TD error analysis."""
        save_dir = save_dir or self.save_dir
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        self._plot_td_abs_analysis(axes[0, :], window)
        self._plot_td_sq_analysis(axes[1, :], window)

        plt.tight_layout()
        if save:
            _save_figure(save_dir, "td_error_analysis.png")

    def _plot_td_abs_analysis(self, axes, window):
        """Plot absolute TD error raw and smoothed."""
        valid_td_abs = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_mean_abs_td_error
        )
        if not valid_td_abs:
            return

        episodes, td_errors = zip(*valid_td_abs)
        axes[0].plot(episodes, td_errors, color="orange")
        axes[0].set_title("Mean Absolute TD Error")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Absolute TD Error")
        axes[0].grid(True)

        if len(td_errors) > window:
            smoothed = np.convolve(td_errors, np.ones(window) / window, mode="valid")
            axes[1].plot(episodes[window - 1:], smoothed, color="orange")
            axes[1].set_title(f"Mean Abs TD Error (Smoothed, window={window})")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Absolute TD Error")
            axes[1].grid(True)

    def _plot_td_sq_analysis(self, axes, window):
        """Plot squared TD error raw and smoothed."""
        valid_td_sq = _filter_valid_data(
            self.metrics.episodes, self.metrics.episode_mean_squared_td_error
        )
        if not valid_td_sq:
            return

        episodes, td_squared = zip(*valid_td_sq)
        axes[0].plot(episodes, td_squared, color="red")
        axes[0].set_title("Mean Squared TD Error")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Squared TD Error")
        axes[0].grid(True)

        if len(td_squared) > window:
            smoothed = np.convolve(td_squared, np.ones(window) / window, mode="valid")
            axes[1].plot(episodes[window - 1:], smoothed, color="red")
            axes[1].set_title(f"Mean Sq TD Error (Smoothed, window={window})")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Squared TD Error")
            axes[1].grid(True)

    def plot_reward_distribution(self, save=False, save_dir=None):
        """Plot histogram of episode rewards."""
        save_dir = save_dir or self.save_dir
        plt.figure(figsize=(10, 6))
        plt.hist(self.metrics.episode_rewards, bins=50, alpha=0.7, edgecolor="black")
        plt.title("Episode Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        if save:
            _save_figure(save_dir, "reward_distribution.png")

    def print_summary_stats(self):
        """Print comprehensive training summary statistics."""
        if not self.metrics.episode_rewards:
            self.logger.warning("No metrics to summarize")
            return

        rewards = np.array(self.metrics.episode_rewards)
        lengths = np.array(self.metrics.episode_steps)

        self._print_basic_stats(rewards, lengths)
        self._print_q_value_stats()
        self._print_loss_stats()
        self._print_td_error_stats()
        self._print_epsilon_stats()
        self._print_recent_stats(rewards, lengths)

    def _print_basic_stats(self, rewards, lengths):
        """Print basic reward and length statistics."""
        self.logger.info("\n=== Training Summary ===")
        self.logger.info(f"Episodes completed: {len(rewards)}")
        self.logger.info(f"Average reward: {rewards.mean():.2f}")
        self.logger.info(f"Best reward: {rewards.max():.2f}")
        self.logger.info(f"Worst reward: {rewards.min():.2f}")
        self.logger.info(f"Reward std: {rewards.std():.2f}")
        self.logger.info(f"Average episode length: {lengths.mean():.1f}")
        self.logger.info(f"Max episode length: {lengths.max()}")
        self.logger.info(f"Min episode length: {lengths.min()}")

    def _print_q_value_stats(self):
        """Print Q-value statistics."""
        valid_q_means = [q for q in self.metrics.episode_q_means if q is not None]
        if not valid_q_means:
            return

        q_means = np.array(valid_q_means)
        self.logger.info(f"\nAverage Q-mean: {q_means.mean():.4f}")
        self.logger.info(f"Max Q-mean: {q_means.max():.4f}")
        self.logger.info(f"Min Q-mean: {q_means.min():.4f}")

    def _print_loss_stats(self):
        """Print loss statistics."""
        valid_losses = [
            loss for loss in self.metrics.episode_losses if loss is not None
        ]
        if valid_losses:
            losses = np.array(valid_losses)
            self.logger.info(f"\nAverage Huber loss: {losses.mean():.6f}")
            self.logger.info(f"Final Huber loss: {losses[-1]:.6f}")

        valid_mse = [mse for mse in self.metrics.episode_mse_losses if mse is not None]
        if valid_mse:
            mse_losses = np.array(valid_mse)
            self.logger.info(f"Average MSE loss: {mse_losses.mean():.6f}")
            self.logger.info(f"Final MSE loss: {mse_losses[-1]:.6f}")

    def _print_td_error_stats(self):
        """Print TD error statistics."""
        valid_td_abs = [
            td for td in self.metrics.episode_mean_abs_td_error if td is not None
        ]
        if valid_td_abs:
            td_abs = np.array(valid_td_abs)
            self.logger.info(f"\nAverage absolute TD error: {td_abs.mean():.6f}")
            self.logger.info(f"Final absolute TD error: {td_abs[-1]:.6f}")

        valid_td_sq = [
            td for td in self.metrics.episode_mean_squared_td_error if td is not None
        ]
        if valid_td_sq:
            td_sq = np.array(valid_td_sq)
            self.logger.info(f"Average squared TD error: {td_sq.mean():.6f}")
            self.logger.info(f"Final squared TD error: {td_sq[-1]:.6f}")

    def _print_epsilon_stats(self):
        """Print epsilon statistics."""
        if self.metrics.episode_epsilons:
            self.logger.info(
                f"\nFinal epsilon: {self.metrics.episode_epsilons[-1]:.4f}"
            )

    def _print_recent_stats(self, rewards, lengths):
        """Print recent episode statistics."""
        if len(rewards) > 100:
            recent_rewards = rewards[-100:]
            recent_lengths = lengths[-100:]
            self.logger.info(
                f"\nRecent 100 episodes avg reward: {recent_rewards.mean():.2f}"
            )
            self.logger.info(
                f"Recent 100 episodes avg length: {recent_lengths.mean():.1f}"
            )
