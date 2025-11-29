import argparse
from dataclasses import dataclass


@dataclass
class GQNConfig:
    """Configuration for Growing Q-Networks."""

    task: str = "walker_walk"
    seed: int = 0
    num_episodes: int = 2000
    max_steps_per_episode: int = 1000

    use_pixels: bool = False
    discount: float = 0.99
    n_step: int = 3
    batch_size: int = 256
    learning_rate: float = 1e-4
    target_update_period: int = 100
    min_replay_size: int = 1000
    max_replay_size: int = 1000000

    initial_bins: int = 2
    final_bins: int = 9
    growing_schedule: str = "adaptive"

    epsilon: float = 0.1
    huber_delta: float = 1.0
    gradient_clip: float = 40.0

    per_alpha: float = 0.6
    per_beta: float = 0.4

    layer_size: int = 512
    num_layers: int = 2

    checkpoint_interval: int = 100
    log_interval: int = 5
    detailed_log_interval: int = 50
    eval_episodes: int = 10

    action_penalty_coeff: float = 0.0

    load_checkpoints: str = None
    load_metrics: str = None


def create_config_from_args(args):
    """Create config from command line arguments."""
    config = GQNConfig()

    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Growing Q-Networks")
    parser.add_argument("--task", type=str, default="walker_walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--use-pixels", action="store_true")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-bins", type=int, default=2)
    parser.add_argument("--final-bins", type=int, default=9)
    parser.add_argument("--growing-schedule", type=str, default="adaptive",
                        choices=["linear", "adaptive"])
    parser.add_argument("--action-penalty-coeff", type=float, default=0.0)
    parser.add_argument("--load-checkpoints", type=str, default=None)
    parser.add_argument("--load-metrics", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=100)

    return parser.parse_args()