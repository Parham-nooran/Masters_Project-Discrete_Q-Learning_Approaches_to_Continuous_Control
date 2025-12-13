import os
import pickle
import json


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
        self.episode_successes = []

        self.bin_selection_per_episode = []
        self.action_range_per_episode = []
        self.q_values_per_level = {i: [] for i in range(10)}
        self.unique_bins_explored = {i: {j: set() for j in range(10)} for i in range(10)}

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
            selected_bins=None,
            action_ranges=None,
            q_values_by_level=None,
            success=None,
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
        self.episode_successes.append(
            success if success is not None else 0.0
        )

        if selected_bins is not None:
            self.bin_selection_per_episode.append({
                'episode': episode,
                'bins': selected_bins
            })

        if action_ranges is not None:
            self.action_range_per_episode.append({
                'episode': episode,
                'ranges': action_ranges
            })

        if q_values_by_level is not None:
            for level, q_val in q_values_by_level.items():
                self.q_values_per_level[level].append(q_val)

    def save_metrics(self, agent, task_name, seed, env_type=None):
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
            "bin_selection_per_episode": self.bin_selection_per_episode,
            "action_range_per_episode": self.action_range_per_episode,
            "q_values_per_level": self.q_values_per_level,
            "env_type": env_type,
        }

        os.makedirs(self.save_dir, exist_ok=True)
        metrics_path = os.path.join(self.save_dir, f"{agent}_{task_name}_{seed}.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics_data, f)

        json_path = os.path.join(self.save_dir, f"{agent}_{task_name}_{seed}_exploration.json")
        exploration_data = {
            "bin_selections": [
                {k: v for k, v in item.items() if k != 'bins'}
                for item in self.bin_selection_per_episode[-100:]
            ],
            "action_ranges": self.action_range_per_episode[-100:],
        }
        with open(json_path, "w") as f:
            json.dump(exploration_data, f, indent=2)

        self.logger.info(f"Metrics saved to {metrics_path}")
        self.logger.info(f"Exploration data saved to {json_path}")

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
                self.episode_successes = metrics_data.get("episode_successes", [])
                self.bin_selection_per_episode = metrics_data.get(
                    "bin_selection_per_episode", []
                )
                self.action_range_per_episode = metrics_data.get(
                    "action_range_per_episode", []
                )
                self.q_values_per_level = metrics_data.get(
                    "q_values_per_level", {i: [] for i in range(10)}
                )
                self.env_type = metrics_data.get("env_type", None)

                self.logger.info(f"Loaded metrics for {len(self.episodes)} episodes")
                return True
            except Exception as e:
                self.logger.warn(f"Failed to load metrics: {e}")
                return False
        return False

    def get_success_rate(self, window=100):
        """Calculate success rate over recent episodes."""
        if not self.episode_successes:
            return 0.0

        recent_successes = self.episode_successes[-window:]
        return sum(recent_successes) / len(recent_successes) if recent_successes else 0.0

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

    def get_exploration_summary(self):
        if not self.bin_selection_per_episode:
            return {}

        recent_selections = self.bin_selection_per_episode[-100:]

        summary = {
            "total_episodes_tracked": len(self.bin_selection_per_episode),
            "recent_episodes": len(recent_selections),
        }

        return summary