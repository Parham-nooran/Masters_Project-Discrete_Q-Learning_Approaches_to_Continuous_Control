import numpy as np
from collections import deque


class MetricsAccumulator:
    """Accumulates recent metrics for logging."""

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.q1_means = deque(maxlen=window_size)
        self.mean_abs_td_errors = deque(maxlen=window_size)
        self.squared_td_errors = deque(maxlen=window_size)
        self.mse_losses = deque(maxlen=window_size)

    def update(self, metrics):
        """Update accumulators with new metrics."""
        if not metrics:
            return

        self.q1_means.append(metrics["q1_mean"])
        self.mse_losses.append(metrics["mse_loss1"])
        self.mean_abs_td_errors.append(metrics["mean_abs_td_error"])
        self.squared_td_errors.append(metrics["mean_squared_td_error"])

        if "loss" in metrics and metrics["loss"] is not None:
            self.losses.append(metrics["loss"])

    def get_averages(self):
        """Get average values of accumulated metrics."""
        return {
            "loss": np.mean(self.losses) if self.losses else 0.0,
            "q_mean": np.mean(self.q1_means) if self.q1_means else 0.0,
            "mean_abs_td_error": (
                np.mean(self.mean_abs_td_errors) if self.mean_abs_td_errors else 0.0
            ),
            "mean_squared_td_error": (
                np.mean(self.squared_td_errors) if self.squared_td_errors else 0.0
            ),
            "mse_loss": np.mean(self.mse_losses) if self.mse_losses else 0.0,
        }
