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

    feature_dim: int = 512
    hidden_dim: int = 1024
    num_levels: int = 3
    num_bins: int = 5

    lr: float = 1e-4
    batch_size: int = 256
    max_episodes: int = 1000
    discount: float = 0.99

    replay_buffer_size: int = 1000000
    min_buffer_size: int = 5000

    num_seed_frames: int = 4000
    action_repeat: int = 2
    update_every_steps: int = 2

    critic_target_tau: float = 0.01
    stddev_schedule: str = "linear(1.0,0.1,100000)"

    num_atoms: int = 51
    v_min: float = 0.0
    v_max: float = 200.0

    eval_frequency: int = 50
    save_frequency: int = 100
    num_eval_episodes: int = 10

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
        if value is not None:
            setattr(config, key.replace("-", "_"), value)
    return config