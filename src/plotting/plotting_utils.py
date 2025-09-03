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
    def __init__(self, metrics_tracker, save_dir="./output/plots"):
        self.metrics = metrics_tracker
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self, window=100, save=False, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        fig, axes = plt.subplots(3, 2, figsize=(15, 18))

        # Rewards
        axes[0, 0].plot(self.metrics.episodes, self.metrics.episode_rewards)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

        # Moving average of rewards
        if len(self.metrics.episode_rewards) > 10:
            window = min(window, len(self.metrics.episode_rewards) // 10)
            moving_avg = np.convolve(
                self.metrics.episode_rewards, np.ones(window) / window, mode="valid"
            )
            axes[0, 1].plot(self.metrics.episodes[window - 1:], moving_avg)
            axes[0, 1].set_title(f"Moving Average Rewards (window={window})")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Average Reward")
            axes[0, 1].grid(True)

        # Losses
        valid_losses = [
            (ep, loss)
            for ep, loss in zip(self.metrics.episodes, self.metrics.episode_losses)
            if loss >= 0
        ]
        if valid_losses:
            episodes, losses = zip(*valid_losses)
            axes[1, 0].plot(episodes, losses)
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            # axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)

        # Q-means
        valid_q_means = [
            (ep, q_mean)
            for ep, q_mean in zip(self.metrics.episodes, self.metrics.episode_q_means)
            if q_mean >= 0
        ]
        if valid_q_means:
            episodes, q_means = zip(*valid_q_means)
            axes[1, 1].plot(episodes, q_means)
            axes[1, 1].set_title("Q-value Means")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Q-mean")
            axes[1, 1].grid(True)

        # Epsilon
        axes[2, 0].plot(self.metrics.episodes, self.metrics.episode_epsilons)
        axes[2, 0].set_title("Epsilon Decay")
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].set_ylabel("Epsilon")
        axes[2, 0].grid(True)



        # Moving average of Q-means
        if valid_q_means and len(q_means) > 10:
            window = min(window, len(q_means) // 10)
            moving_avg_q = np.convolve(q_means, np.ones(window) / window, mode="valid")
            axes[2, 1].plot(episodes[window - 1:], moving_avg_q)
            axes[2, 1].set_title(f"Moving Average Q-means (window={window})")
            axes[2, 1].set_xlabel("Episode")
            axes[2, 1].set_ylabel("Average Q-mean")
            axes[2, 1].grid(True)



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
        lengths = np.array(self.metrics.episode_lengths)

        print("\n=== Training Summary ===")
        print(f"Episodes completed: {len(rewards)}")
        print(f"Average reward: {rewards.mean():.2f}")
        print(f"Best reward: {rewards.max():.2f}")
        print(f"Worst reward: {rewards.min():.2f}")
        print(f"Reward std: {rewards.std():.2f}")
        print(f"Average episode length: {lengths.mean():.1f}")
        print(f"Max episode length: {lengths.max()}")
        print(f"Min episode length: {lengths.min()}")

        # Q-means stats
        valid_q_means = [q for q in self.metrics.episode_q_means if q >= 0]
        if valid_q_means:
            q_means = np.array(valid_q_means)
            print(f"Average Q-mean: {q_means.mean():.4f}")
            print(f"Max Q-mean: {q_means.max():.4f}")
            print(f"Min Q-mean: {q_means.min():.4f}")

        # Loss stats
        valid_losses = [loss for loss in self.metrics.episode_losses if loss >= 0]
        if valid_losses:
            losses = np.array(valid_losses)
            print(f"Average loss: {losses.mean():.6f}")
            print(f"Final epsilon: {self.metrics.episode_epsilons[-1]:.4f}")

        if len(rewards) > 100:
            recent_rewards = rewards[-100:]
            print(f"Recent 100 episodes avg reward: {recent_rewards.mean():.2f}")
            recent_lengths = lengths[-100:]
            print(f"Recent 100 episodes avg length: {recent_lengths.mean():.1f}")