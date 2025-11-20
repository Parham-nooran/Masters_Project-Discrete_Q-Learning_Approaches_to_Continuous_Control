import numpy as np
from collections import deque


class GrowingScheduler:
    """Conservative adaptive scheduler to prevent premature growth."""

    def __init__(self, total_episodes, window_size=100, min_episodes_between_growth=150):
        self.total_episodes = total_episodes
        self.window_size = window_size
        self.min_episodes_between_growth = min_episodes_between_growth

        self.returns_history = deque(maxlen=window_size)
        self.last_growth_episode = 0

    def should_grow(self, episode, episode_return):
        """Determine if action space should grow based on performance plateau."""
        self.returns_history.append(episode_return)

        if not self._can_grow(episode):
            return False

        if not self._has_plateaued():
            return False

        self.last_growth_episode = episode
        return True

    def _can_grow(self, episode):
        """Check if enough episodes have passed to consider growth."""
        return (
                len(self.returns_history) >= self.window_size
                and episode - self.last_growth_episode >= self.min_episodes_between_growth
        )

    def _has_plateaued(self):
        """Check if performance has plateaued using conservative threshold."""
        recent_window_size = 20

        if len(self.returns_history) < self.window_size:
            return False

        recent_returns = list(self.returns_history)[-recent_window_size:]
        earlier_returns = list(self.returns_history)[:-recent_window_size]

        if len(earlier_returns) < recent_window_size:
            return False

        recent_mean = np.mean(recent_returns)
        earlier_mean = np.mean(earlier_returns)

        improvement_threshold = 0.05 * max(abs(earlier_mean), 1.0)
        actual_improvement = recent_mean - earlier_mean

        return actual_improvement < improvement_threshold