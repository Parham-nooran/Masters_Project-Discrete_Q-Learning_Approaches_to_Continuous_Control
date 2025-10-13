import os
import pickle


class MetricsTracker:
    def __init__(self, save_dir="./metrics"):
        self.save_dir = save_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_mean_abs_td_error = []
        self.episode_mean_squared_td_error = []
        self.episode_q_means = []
        self.episode_epsilons = []
        self.episode_mse_losses = []
        self.episode_times = []
        self.episodes = []

        os.makedirs(save_dir, exist_ok=True)

    def log_episode(
        self,
        episode,
        reward,
        length,
        mse_loss=0.0,
        loss=0.0,
        mean_abs_td_error=0.0,
        mean_squared_td_error=0.0,
        q_mean=0.0,
        epsilon=0.0,
        episode_time=0.0,
    ):
        self.episodes.append(episode)
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_losses.append(loss)
        self.episode_mean_abs_td_error.append(mean_abs_td_error)
        self.episode_mean_squared_td_error.append(mean_squared_td_error)
        self.episode_q_means.append(q_mean)
        self.episode_epsilons.append(epsilon)
        self.episode_mse_losses.append(mse_loss)
        self.episode_times.append(episode_time)

    def save_metrics(self):
        metrics_data = {
            "episodes": self.episodes,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_losses": self.episode_losses,
            "episode_mse_losses": self.episode_mse_losses,
            "episode_mean_abs_td_error": self.episode_mean_abs_td_error,
            "episode_mean_squared_td_error": self.episode_mean_squared_td_error,
            "episode_epsilons": self.episode_epsilons,
            "episode_q_means": self.episode_q_means,
            "episode_times": self.episode_times,
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
                self.episode_mse_losses = metrics_data.get("episode_mse_losses", [])
                self.episode_losses = metrics_data.get("episode_losses", [])
                self.episode_mean_abs_td_error = metrics_data.get(
                    "episode_mean_abs_td_error", []
                )
                self.episode_mean_squared_td_error = metrics_data.get(
                    "episode_mean_squared_td_error", []
                )
                self.episode_q_means = metrics_data.get("episode_q_means", [])
                self.episode_epsilons = metrics_data.get("episode_epsilons", [])

                print(f"Loaded metrics for {len(self.episodes)} episodes")
                return True
            except Exception as e:
                print(f"Failed to load metrics: {e}")
                return False
        return False
