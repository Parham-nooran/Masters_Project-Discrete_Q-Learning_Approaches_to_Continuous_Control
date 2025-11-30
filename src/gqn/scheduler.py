import numpy as np
from collections import deque


class GrowthScheduler:
    """Manages when to grow action space resolution."""

    def __init__(self, schedule_type, num_episodes, num_growth_stages):
        self.schedule_type = schedule_type
        self.num_episodes = num_episodes
        self.num_growth_stages = num_growth_stages

        if schedule_type == "linear":
            self.growth_episodes = self._compute_linear_schedule()
        else:
            self.growth_episodes = []
            self.return_history = deque(maxlen=100)
            self.moving_avg_mean = 0.0
            self.moving_avg_std = 0.0
            self.last_growth_episode = -1
            self.min_episodes_between_growth = 50

    def _compute_linear_schedule(self):
        """Compute episodes at which to grow for linear schedule."""
        if self.num_growth_stages <= 1:
            return []

        interval = self.num_episodes / self.num_growth_stages
        return [int(interval * (i + 1)) for i in range(self.num_growth_stages - 1)]

    def should_grow(self, episode, current_return=None):
        """Determine if action space should grow."""
        if self.schedule_type == "linear":
            return episode in self.growth_episodes
        else:
            return self._should_grow_adaptive(episode, current_return)

    def _should_grow_adaptive(self, episode, current_return):
        """Adaptive growth based on performance stagnation."""
        if current_return is None:
            return False

        if episode - self.last_growth_episode < self.min_episodes_between_growth:
            return False

        self.return_history.append(current_return)

        if len(self.return_history) < 10:
            return False

        returns = np.array(self.return_history)
        self.moving_avg_mean = np.mean(returns)
        self.moving_avg_std = np.std(returns)

        threshold = self._compute_threshold()

        should_grow = self.moving_avg_mean < threshold

        if should_grow:
            self.last_growth_episode = episode

        return should_grow

    def _compute_threshold(self):
        """Compute threshold for adaptive growth."""
        sign = np.sign(self.moving_avg_mean) if self.moving_avg_mean != 0 else 1
        threshold = (1.0 - 0.02 * sign) * self.moving_avg_mean + 0.5 * self.moving_avg_std
        return threshold

    def get_status(self):
        """Get current scheduler status."""
        if self.schedule_type == "linear":
            return {
                "type": "linear",
                "growth_episodes": self.growth_episodes
            }
        else:
            return {
                "type": "adaptive",
                "moving_avg_mean": self.moving_avg_mean,
                "moving_avg_std": self.moving_avg_std,
                "threshold": self._compute_threshold() if len(self.return_history) >= 10 else None,
                "history_size": len(self.return_history),
                "last_growth_episode": self.last_growth_episode
            }