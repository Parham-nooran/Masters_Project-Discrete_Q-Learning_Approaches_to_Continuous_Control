import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class MetricsTracker:
    def __init__(self, save_dir="./metrics"):
        self.save_dir = save_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_q_means = []
        self.episode_epsilons = []
        self.episodes = []

        os.makedirs(save_dir, exist_ok=True)

    def log_episode(
        self, episode, reward, length, loss=None, q_mean=None, epsilon=None
    ):
        self.episodes.append(episode)
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_losses.append(loss if loss is not None else 0.0)
        self.episode_q_means.append(q_mean if q_mean is not None else 0.0)
        self.episode_epsilons.append(epsilon if epsilon is not None else 0.0)

    def save_metrics(self):
        metrics_data = {
            "episodes": self.episodes,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_losses": self.episode_losses,
            "episode_q_means": self.episode_q_means,
            "episode_epsilons": self.episode_epsilons,
        }

        metrics_path = os.path.join(self.save_dir, "metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics_data, f)

    def load_metrics(self, path=None):
        if path is None:
            path = os.path.join(self.save_dir, "metrics.pkl")

        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    metrics_data = pickle.load(f)

                self.episodes = metrics_data.get("episodes", [])
                self.episode_rewards = metrics_data.get("episode_rewards", [])
                self.episode_lengths = metrics_data.get("episode_lengths", [])
                self.episode_losses = metrics_data.get("episode_losses", [])
                self.episode_q_means = metrics_data.get("episode_q_means", [])
                self.episode_epsilons = metrics_data.get("episode_epsilons", [])

                print(f"Loaded metrics for {len(self.episodes)} episodes")
                return True
            except Exception as e:
                print(f"Failed to load metrics: {e}")
                return False
        return False


class PlottingUtils:
    def __init__(self, metrics_tracker, save_dir="./plots"):
        self.metrics = metrics_tracker
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self, window=100, save=False, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Rewards
        axes[0, 0].plot(self.metrics.episodes, self.metrics.episode_rewards)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        # Moving average of rewards
        if len(self.metrics.episode_rewards) > 10:
            window = min(window, len(self.metrics.episode_rewards) // 10)
            moving_avg = np.convolve(
                self.metrics.episode_rewards, np.ones(window) / window, mode="valid"
            )
            axes[0, 1].plot(self.metrics.episodes[window - 1 :], moving_avg)
            axes[0, 1].set_title(f"Moving Average Rewards (window={window})")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Average Reward")

        # Losses
        valid_losses = [
            (ep, loss)
            for ep, loss in zip(self.metrics.episodes, self.metrics.episode_losses)
            if loss > 0
        ]
        if valid_losses:
            episodes, losses = zip(*valid_losses)
            axes[1, 0].plot(episodes, losses)
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")

        # Epsilon
        axes[1, 1].plot(self.metrics.episodes, self.metrics.episode_epsilons)
        axes[1, 1].set_title("Epsilon Decay")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Epsilon")

        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(save_dir, "training_curves.png"),
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

    def plot_reward_distribution(self, save=False, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        plt.figure(figsize=(10, 6))
        plt.hist(self.metrics.episode_rewards, bins=50, alpha=0.7, edgecolor="black")
        plt.title("Episode Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")

        if save:
            plt.savefig(
                os.path.join(save_dir, "reward_distribution.png"),
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

    def print_summary_stats(self):
        if not self.metrics.episode_rewards:
            print("No metrics to summarize")
            return

        rewards = np.array(self.metrics.episode_rewards)
        print("\n=== Training Summary ===")
        print(f"Episodes completed: {len(rewards)}")
        print(f"Average reward: {rewards.mean():.2f}")
        print(f"Best reward: {rewards.max():.2f}")
        print(f"Worst reward: {rewards.min():.2f}")
        print(f"Reward std: {rewards.std():.2f}")

        if len(rewards) > 100:
            recent_rewards = rewards[-100:]
            print(f"Recent 100 episodes avg: {recent_rewards.mean():.2f}")
