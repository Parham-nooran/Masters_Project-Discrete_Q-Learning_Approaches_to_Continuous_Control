import numpy as np
from collections import deque


class GrowthScheduler:
    """Manages when to grow action space resolution using linear or adaptive scheduling."""

    def __init__(self, schedule_type, num_episodes, num_growth_stages):
        self.schedule_type = schedule_type
        self.num_episodes = num_episodes
        self.num_growth_stages = num_growth_stages

        if schedule_type == "linear":
            self.growth_episodes = self._compute_linear_schedule()
        else:
            self._initialize_adaptive_scheduler()

    def _initialize_adaptive_scheduler(self):
        """Initialize parameters for adaptive scheduling."""
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
        """Determine if action space should grow based on schedule type."""
        if self.schedule_type == "linear":
            return episode in self.growth_episodes
        return self._should_grow_adaptive(episode, current_return)

    def _should_grow_adaptive(self, episode, current_return):
        """Adaptive growth based on performance stagnation."""
        if not self._is_valid_for_adaptive_growth(episode, current_return):
            return False

        self.return_history.append(current_return)

        if not self._has_sufficient_history():
            return False

        self._update_moving_statistics()

        if self._is_performance_stagnating():
            self.last_growth_episode = episode
            return True

        return False

    def _is_valid_for_adaptive_growth(self, episode, current_return):
        """Check if conditions are met for considering adaptive growth."""
        if current_return is None:
            return False
        return episode - self.last_growth_episode >= self.min_episodes_between_growth

    def _has_sufficient_history(self):
        """Check if enough episodes have been collected for reliable statistics."""
        return len(self.return_history) >= 50

    def _update_moving_statistics(self):
        """Update moving average mean and standard deviation from return history."""
        returns = np.array(self.return_history)
        self.moving_avg_mean = np.mean(returns)
        self.moving_avg_std = np.std(returns)

    def _is_performance_stagnating(self):
        """Check if current performance is below threshold indicating stagnation."""
        threshold = self._compute_threshold()
        return self.moving_avg_mean < threshold

    def _compute_threshold(self):
        """Compute threshold for adaptive growth (Paper Equation 4)."""
        sign = np.sign(self.moving_avg_mean) if self.moving_avg_mean != 0 else 1
        threshold = (
            1.0 - 0.05 * sign
        ) * self.moving_avg_mean + 0.90 * self.moving_avg_std
        return threshold

    def get_status(self):
        """Get current scheduler status."""
        if self.schedule_type == "linear":
            return self._get_linear_status()
        return self._get_adaptive_status()

    def _get_linear_status(self):
        """Get status for linear scheduler."""
        return {"type": "linear", "growth_episodes": self.growth_episodes}

    def _get_adaptive_status(self):
        """Get status for adaptive scheduler."""
        return {
            "type": "adaptive",
            "moving_avg_mean": self.moving_avg_mean,
            "moving_avg_std": self.moving_avg_std,
            "threshold": (
                self._compute_threshold() if len(self.return_history) >= 10 else None
            ),
            "history_size": len(self.return_history),
            "last_growth_episode": self.last_growth_episode,
        }
