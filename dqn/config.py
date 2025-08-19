from typing import List
from dataclasses import dataclass


@dataclass
class Config:
    # General
    task: str = "walker_walk"
    num_episodes: int = 1000
    num_steps: int = 1000000
    seed: int = 0
    mock_episode_length: int = 100  # Added for mock training

    # Algorithm
    algorithm: str = 'dqn'
    num_bins: int = 3  # Changed to 3 for bang-off-bang control (paper default)

    # Network
    layer_size_network: List[int] = None
    learning_rate: float = 1e-4
    clip_gradients: bool = True
    clip_gradients_norm: float = 40.0  # Paper specification
    epsilon: float = 0.1

    # RL hyperparams
    discount: float = 0.99
    batch_size: int = 256
    target_update_period: int = 100
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: float = 32.0

    # Prioritized replay
    importance_sampling_exponent: float = 0.2
    priority_exponent: float = 0.6

    # Double Q and architecture
    use_double_q: bool = True  # Changed to True (DecQN uses Double Q-learning)
    use_residual: bool = False

    # Vision
    num_pixels: int = 84
    pad_size: int = 4
    layer_size_bottleneck: int = 100

    # N-step
    adder_n_step: int = 3

    # Huber loss (added for DecQN)
    huber_loss_parameter: float = 1.0

    # Derived properties
    @property
    def decouple(self) -> bool:
        return 'decqn' in self.algorithm

    @property
    def use_pixels(self) -> bool:
        return 'vis' in self.algorithm

    def __post_init__(self):
        if self.layer_size_network is None:
            self.layer_size_network = [512, 512]

        # Adjust network size for vision tasks
        if self.use_pixels:
            self.layer_size_network = [1024, 1024]

    def get(self, key: str, default=None):
        """Get configuration value with default fallback (for backward compatibility)."""
        return getattr(self, key, default)

    def set_decqn_defaults(self):
        """Set DecQN-specific defaults based on paper specifications."""
        self.algorithm = 'decqn'
        self.use_double_q = True
        self.num_bins = 3
        self.clip_gradients = True
        self.clip_gradients_norm = 40.0

    def set_decqn_vision_defaults(self):
        """Set DecQN vision-specific defaults."""
        self.algorithm = 'decqnvis'
        self.use_double_q = True
        self.num_bins = 2  # Bang-bang for vision tasks
        self.clip_gradients = True
        self.clip_gradients_norm = 40.0