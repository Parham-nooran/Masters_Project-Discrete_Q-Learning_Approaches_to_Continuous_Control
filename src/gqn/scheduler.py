import numpy as np
from collections import deque

class GrowingScheduler:
    """
    Implements both linear and adaptive growing schedules from the paper.
    """

    def __init__(self, total_episodes: int, num_growth_stages: int,
                 schedule_type: str = "adaptive", window_size: int = 100):
        self.total_episodes = total_episodes
        self.num_growth_stages = num_growth_stages
        self.schedule_type = schedule_type
        self.window_size = window_size

        # For linear schedule
        if schedule_type == "linear":
            self.growth_episodes = [
                int(total_episodes * (i + 1) / (num_growth_stages + 1))
                for i in range(num_growth_stages)
            ]

        # For adaptive schedule
        self.returns_history = deque(maxlen=window_size)
        self.last_growth_episode = 0
        self.min_episodes_between_growth = total_episodes // (num_growth_stages * 3)  # Prevent too frequent growth

    def should_grow(self, episode: int, episode_return: float) -> bool:
        """
        Determine if action space should grow based on schedule.
        """
        if self.schedule_type == "linear":
            return episode in self.growth_episodes

        elif self.schedule_type == "adaptive":
            self.returns_history.append(episode_return)

            # Need minimum history and minimum episodes since last growth
            if (len(self.returns_history) < self.window_size or
                    episode - self.last_growth_episode < self.min_episodes_between_growth):
                return False

            # Implement Equation 4 from paper
            mu_ma = np.mean(self.returns_history)
            sigma_ma = np.std(self.returns_history)

            # Paper's Equation 4: G_threshold = (1 - 0.05 * sign(μ_MA)) * μ_MA + 0.90 * σ_MA
            sign_mu = np.sign(mu_ma) if mu_ma != 0 else 0
            threshold = (1.0 - 0.05 * sign_mu) * mu_ma + 0.90 * sigma_ma

            # Growth occurs when performance stagnates below threshold
            should_grow = episode_return < threshold

            if should_grow:
                self.last_growth_episode = episode

            return should_grow

        return False

