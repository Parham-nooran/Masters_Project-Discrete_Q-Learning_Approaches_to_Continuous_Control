import os
import pickle


class MetricsTracker:
    def __init__(self, logger, save_dir):
        self.logger = logger
        self.save_dir = save_dir
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_losses = []
        self.episode_mean_abs_td_error = []
        self.episode_mean_squared_td_error = []
        self.episode_q_means = []
        self.episode_epsilons = []
        self.episode_mse_losses = []
        self.episode_times = []
        self.episodes = []
        self.episode_bin_widths = []
        self.episode_current_bins = []
        self.episode_growth_history = []
        os.makedirs(save_dir, exist_ok=True)

    def log_episode(
        self,
        episode,
        reward,
        steps,
        mse_loss=0.0,
        loss=0.0,
        mean_abs_td_error=0.0,
        mean_squared_td_error=0.0,
        q_mean=0.0,
        epsilon=0.0,
        episode_time=0.0,
        current_bins=None,
        growth_history=None,
        **kwargs,
    ):
        self.episodes.append(episode)
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_losses.append(loss)
        self.episode_mean_abs_td_error.append(mean_abs_td_error)
        self.episode_mean_squared_td_error.append(mean_squared_td_error)
        self.episode_q_means.append(q_mean)
        self.episode_epsilons.append(epsilon)
        self.episode_mse_losses.append(mse_loss)
        self.episode_times.append(episode_time)
        self.episode_current_bins.append(
            current_bins if current_bins is not None else 0
        )
        self.episode_growth_history.append(
            growth_history if growth_history is not None else "[]"
        )

    def save_metrics(self, agent, task_name):
        metrics_data = {
            "episodes": self.episodes,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "episode_losses": self.episode_losses,
            "episode_mse_losses": self.episode_mse_losses,
            "episode_mean_abs_td_error": self.episode_mean_abs_td_error,
            "episode_mean_squared_td_error": self.episode_mean_squared_td_error,
            "episode_epsilons": self.episode_epsilons,
            "episode_q_means": self.episode_q_means,
            "episode_times": self.episode_times,
            "episode_bin_widths": self.episode_bin_widths,
            "episode_current_bins": self.episode_current_bins,
            "episode_growth_history": self.episode_growth_history,
        }

        os.makedirs(self.save_dir, exist_ok=True)
        metrics_path = os.path.join(self.save_dir, f"{agent}_{task_name}.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics_data, f)

        self.logger.info(f"Metrics saved to {metrics_path}")

    def load_metrics(self, path=None):
        if path is None:
            path = os.path.join(self.save_dir, "metrics.pkl")

        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    metrics_data = pickle.load(f)

                self.episodes = metrics_data.get("episodes", [])
                self.episode_rewards = metrics_data.get("episode_rewards", [])
                self.episode_steps = metrics_data.get("episode_steps", [])
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
                self.episode_times = metrics_data.get("episode_times", [])
                self.episode_bin_widths = metrics_data.get("episode_bin_widths", [])
                self.episode_current_bins = metrics_data.get("episode_current_bins", [])
                self.episode_growth_history = metrics_data.get(
                    "episode_growth_history", []
                )

                self.logger.info(f"Loaded metrics for {len(self.episodes)} episodes")
                return True
            except Exception as e:
                self.logger.warn(f"Failed to load metrics: {e}")
                return False
        return False

    def get_growth_events(self):
        """Extract growth events from history."""
        growth_events = []
        prev_bins = None

        for episode, bins_str in zip(self.episodes, self.episode_growth_history):
            try:
                bins_list = eval(bins_str) if isinstance(bins_str, str) else bins_str
                if bins_list and (prev_bins is None or bins_list != prev_bins):
                    current_bins = bins_list[-1] if bins_list else None
                    if current_bins and current_bins != prev_bins:
                        growth_events.append(
                            {
                                "episode": episode,
                                "bins": current_bins,
                                "full_history": bins_list,
                            }
                        )
                        prev_bins = current_bins
            except:
                continue

        return growth_events
