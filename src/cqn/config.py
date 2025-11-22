"""
Configuration for Coarse-to-Fine Q-Network agent.
"""

import argparse
from dataclasses import dataclass


@dataclass
class CQNConfig:
    """CQN agent and training hyperparameters."""

    env_name: str = "walker_walk"
    task: str = "walker_walk"
    seed: int = 42

    layer_size_bottleneck: int = 512
    num_levels: int = 2
    num_bins: int = 3

    lr: float = 1e-3
    batch_size: int = 512
    max_episodes: int = 1000
    discount: float = 0.99

    initial_epsilon: float = 1.0
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.995

    replay_buffer_size: int = 1000000
    min_buffer_size: int = 5000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    n_step: int = 3

    target_update_freq: int = 1000
    target_update_tau: float = 0.005
    huber_loss_parameter: float = 1.0
    max_grad_norm: float = 10.0

    eval_frequency: int = 50
    save_frequency: int = 100

    log_level: str = "INFO"
    load_checkpoints: str = None


def create_config(args: argparse.Namespace) -> CQNConfig:
    """
    Create CQNConfig from parsed arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        CQNConfig instance.
    """
    config = CQNConfig()
    for key, value in vars(args).items():
        setattr(config, key.replace("-", "_"), value)
    return config
