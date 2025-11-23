import argparse
from dataclasses import dataclass


@dataclass
class CQNConfig:
    env_name: str = "walker_walk"
    task: str = "walker_walk"
    seed: int = 42

    layer_size_bottleneck: int = 512
    num_levels: int = 3
    num_bins: int = 5

    lr: float = 5e-5
    weight_decay: float = 0.1
    batch_size: int = 512
    max_episodes: int = 1000
    discount: float = 0.99

    replay_buffer_size: int = 1000000
    min_buffer_size: int = 5000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    n_step: int = 3

    target_update_freq: int = 1
    target_update_tau: float = 0.02
    max_grad_norm: float = 10.0

    num_atoms: int = 51
    v_min: float = -1.0
    v_max: float = 1.0

    eval_frequency: int = 50
    save_frequency: int = 100

    log_level: str = "INFO"
    load_checkpoints: str = None


def create_config(args: argparse.Namespace) -> CQNConfig:
    config = CQNConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key.replace("-", "_"), value)
    return config