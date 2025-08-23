import json
from pathlib import Path
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import floating


class MetricsTracker:
    """Track and save training metrics during DecQN training."""

    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.q_values = []
        self.epsilon_values = []
        self.training_steps = []
        self.episodes = []

    def log_episode(self, episode: int, reward: float, length: int,
                    loss: float = None, q_mean: float = None, epsilon: float = None):
        """Log metrics for a completed episode."""
        self.episodes.append(episode)
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        if loss is not None:
            self.losses.append(loss)
        if q_mean is not None:
            self.q_values.append(q_mean)
        if epsilon is not None:
            self.epsilon_values.append(epsilon)

    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save all metrics to JSON file."""
        metrics = {
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'q_values': self.q_values,
            'epsilon_values': self.epsilon_values,
        }

        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def load_metrics(self, filename: str = "training_metrics.json"):
        """Load metrics from JSON file."""
        filepath = self.save_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                metrics = json.load(f)

            self.episodes = metrics.get('episodes', [])
            self.episode_rewards = metrics.get('episode_rewards', [])
            self.episode_lengths = metrics.get('episode_lengths', [])
            self.losses = metrics.get('losses', [])
            self.q_values = metrics.get('q_values', [])
            self.epsilon_values = metrics.get('epsilon_values', [])

    def get_running_average(self, values: List[float], window: int = 100) -> list[floating[Any]] | list[Any]:
        """Calculate running average with given window size."""
        if len(values) < window:
            return [np.mean(values[:i + 1]) for i in range(len(values))]

        running_avg = []
        for i in range(len(values)):
            if i < window:
                running_avg.append(np.mean(values[:i + 1]))
            else:
                running_avg.append(np.mean(values[i - window + 1:i + 1]))
        return running_avg


class PlottingUtils:
    """Utilities for plotting DecQN training metrics."""

    def __init__(self, metrics_tracker: MetricsTracker, save_dir: str = "plots"):
        self.tracker = metrics_tracker
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def plot_training_curves(self, window: int = 100, save: bool = True):
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DecQN Training Metrics', fontsize=16)

        # Episode Rewards
        if self.tracker.episode_rewards:
            ax = axes[0, 0]
            episodes = self.tracker.episodes
            rewards = self.tracker.episode_rewards
            running_avg = self.tracker.get_running_average(rewards, window)

            ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
            ax.plot(episodes, running_avg, color='red', linewidth=2, label=f'Running Avg ({window})')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Episode Rewards')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Losses
        if self.tracker.losses:
            ax = axes[0, 1]
            loss_episodes = self.tracker.episodes[:len(self.tracker.losses)]
            ax.plot(loss_episodes, self.tracker.losses, color='orange', alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)

        # Q-values
        if self.tracker.q_values:
            ax = axes[1, 0]
            q_episodes = self.tracker.episodes[:len(self.tracker.q_values)]
            ax.plot(q_episodes, self.tracker.q_values, color='green', alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Q-value')
            ax.set_title('Q-value Evolution')
            ax.grid(True, alpha=0.3)

        # Episode Lengths
        if self.tracker.episode_lengths:
            ax = axes[1, 1]
            ax.plot(self.tracker.episodes, self.tracker.episode_lengths, color='purple', alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Episode Length')
            ax.set_title('Episode Lengths')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.save_dir / 'training_curves.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")

        plt.show()

    def plot_reward_distribution(self, save: bool = True):
        """Plot reward distribution histogram."""
        if not self.tracker.episode_rewards:
            print("No reward data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(self.tracker.episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Distribution of Episode Rewards')
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_reward = np.mean(self.tracker.episode_rewards)
        std_reward = np.std(self.tracker.episode_rewards)
        plt.axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        plt.axvline(mean_reward + std_reward, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_reward:.2f}')
        plt.axvline(mean_reward - std_reward, color='orange', linestyle='--', alpha=0.7)
        plt.legend()

        if save:
            filepath = self.save_dir / 'reward_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")

        plt.show()

    def print_summary_stats(self):
        """Print summary statistics."""
        if not self.tracker.episode_rewards:
            print("No data to summarize")
            return

        print("\n" + "=" * 50)
        print("TRAINING SUMMARY STATISTICS")
        print("=" * 50)

        # Reward stats
        rewards = np.array(self.tracker.episode_rewards)
        print(f"Episodes Completed: {len(rewards)}")
        print(f"Mean Reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
        print(f"Max Reward: {rewards.max():.2f}")
        print(f"Min Reward: {rewards.min():.2f}")

        # Recent performance (last 100 episodes)
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            print(f"Recent Mean (last 100): {recent_rewards.mean():.2f} ± {recent_rewards.std():.2f}")

        # Loss stats
        if self.tracker.losses:
            losses = np.array(self.tracker.losses)
            print(f"Final Loss: {losses[-1]:.6f}")
            print(f"Mean Loss: {losses.mean():.6f}")

        # Q-value stats
        if self.tracker.q_values:
            q_vals = np.array(self.tracker.q_values)
            print(f"Final Q-value: {q_vals[-1]:.3f}")
            print(f"Mean Q-value: {q_vals.mean():.3f}")

        print("=" * 50)


def plot_from_saved_metrics(metrics_file: str = "metrics/training_metrics.json"):
    """Load saved metrics and create plots."""
    print("Loading saved metrics and creating plots...")

    tracker = MetricsTracker()
    tracker.load_metrics(metrics_file)

    if not tracker.episode_rewards:
        print(f"No metrics found in {metrics_file}")
        return

    plotter = PlottingUtils(tracker)

    # Create all plots
    plotter.plot_training_curves(window=100, save=True)
    plotter.plot_reward_distribution(save=True)
    plotter.print_summary_stats()

    print("All plots created and saved!")
