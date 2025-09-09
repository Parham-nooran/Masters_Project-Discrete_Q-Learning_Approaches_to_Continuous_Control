import numpy as np
from collections import deque


class GrowingScheduler:
    """Minimal growing scheduler - adaptive only as per 2024 paper."""

    def __init__(self, total_episodes, window_size=100):
        self.total_episodes = total_episodes
        self.window_size = window_size

        # Adaptive schedule parameters
        self.returns_history = deque(maxlen=window_size)
        self.last_growth_episode = 0
        self.min_episodes_between_growth = 100  # Minimum episodes between growth

    def should_grow(self, episode, episode_return):
        """Determine if action space should grow based on adaptive schedule."""
        self.returns_history.append(episode_return)

        # Need minimum history and minimum episodes since last growth
        if (len(self.returns_history) < self.window_size or
                episode - self.last_growth_episode < self.min_episodes_between_growth):
            return False

        # Check if performance has plateaued (simplified criterion)
        recent_returns = list(self.returns_history)[-20:]
        earlier_returns = list(self.returns_history)[:-20]

        if len(earlier_returns) < 20:
            return False

        recent_mean = np.mean(recent_returns)
        earlier_mean = np.mean(earlier_returns)
        improvement = recent_mean - earlier_mean

        # Growth criterion: improvement is less than 5% of earlier performance
        should_grow = improvement < 0.05 * abs(earlier_mean)

        if should_grow:
            self.last_growth_episode = episode

        return should_grow