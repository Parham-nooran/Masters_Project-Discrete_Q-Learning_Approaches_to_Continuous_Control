import os
from dataclasses import dataclass


@dataclass
class CQNConfig:
    """Configuration for CQN agent"""
    env_name: str = "walker_run"
    seed: int = 42

    layer_size_bottleneck: int = 512
    num_levels: int = 3
    num_bins: int = 5

    lr: float = 1e-3
    batch_size: int = 256
    max_episodes: int = 1000
    discount: float = 0.99

    initial_epsilon: float = 1.0
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.995

    replay_buffer_size: int = 1000000
    min_buffer_size: int = 10000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    n_step: int = 3

    target_update_freq: int = 1000
    huber_loss_parameter: float = 1.0
    max_grad_norm: float = 10.0

    eval_frequency: int = 50
    save_frequency: int = 100

    working_dir: str = "experiments"
    save_dir: str = "models/cqn"
    log_level: str = "INFO"

    def __post_init__(self):
        """Post-initialization setup"""
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.working_dir, "output/logs"), exist_ok=True)
