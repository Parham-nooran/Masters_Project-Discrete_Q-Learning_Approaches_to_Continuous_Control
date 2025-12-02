import argparse
from dataclasses import dataclass


@dataclass
class GQNConfig:
    """Configuration for Growing Q-Networks with all hyperparameters."""

    env_type: str = "ogbench"
    task: str = "walker_walk"
    seed: int = 0
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000

    use_pixels: bool = False
    ogbench_dataset_dir: str = "~/.ogbench/data"
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
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01

    huber_delta: float = 1.0
    gradient_clip: float = 40.0

    per_alpha: float = 0.6
    per_beta: float = 0.4

    layer_size: int = 512
    layer_size_bottleneck: int = 50
    num_layers: int = 2

    checkpoint_interval: int = 100
    metrics_save_interval: int = 100
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
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    return config


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Growing Q-Networks")

    _add_environment_arguments(parser)
    _add_training_arguments(parser)
    _add_action_space_arguments(parser)
    _add_exploration_arguments(parser)
    _add_optimization_arguments(parser)
    _add_replay_buffer_arguments(parser)
    _add_network_arguments(parser)
    _add_logging_arguments(parser)
    _add_checkpoint_arguments(parser)

    return parser.parse_args()


def _add_environment_arguments(parser):
    """Add environment-related arguments."""
    parser.add_argument("--env-type", type=str, default="dmcontrol",
                        choices=["dmcontrol", "ogbench"],
                        help="Environment type (dmcontrol or ogbench)")
    parser.add_argument("--task", type=str, default="walker_walk",
                        help="Environment task")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num-episodes", type=int, default=1000,
                        help="Number of episodes to train")
    parser.add_argument("--max-steps-per-episode", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--use-pixels", action="store_true",
                        help="Use pixel observations")
    parser.add_argument("--ogbench-dataset-dir", type=str, default="~/.ogbench/data",
                        help="Directory for OGBench datasets (only for ogbench env type)")


def _add_training_arguments(parser):
    """Add training hyperparameter arguments."""
    parser.add_argument("--discount", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--n-step", type=int, default=3,
                        help="N-step returns")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--target-update-period", type=int, default=100,
                        help="Target network update period")


def _add_action_space_arguments(parser):
    """Add action space growth arguments."""
    parser.add_argument("--initial-bins", type=int, default=2,
                        help="Initial number of bins")
    parser.add_argument("--final-bins", type=int, default=9,
                        help="Final number of bins")
    parser.add_argument("--growing-schedule", type=str, default="adaptive",
                        choices=["linear", "adaptive"],
                        help="Growth schedule type")


def _add_exploration_arguments(parser):
    """Add exploration strategy arguments."""
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Epsilon decay rate")
    parser.add_argument("--min-epsilon", type=float, default=0.01,
                        help="Minimum epsilon")


def _add_optimization_arguments(parser):
    """Add optimization-related arguments."""
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber loss delta")
    parser.add_argument("--gradient-clip", type=float, default=40.0,
                        help="Gradient clipping value")
    parser.add_argument("--action-penalty-coeff", type=float, default=0.0,
                        help="Action penalty coefficient")


def _add_replay_buffer_arguments(parser):
    """Add replay buffer configuration arguments."""
    parser.add_argument("--min-replay-size", type=int, default=1000,
                        help="Minimum replay buffer size before training")
    parser.add_argument("--max-replay-size", type=int, default=1000000,
                        help="Maximum replay buffer size")
    parser.add_argument("--per-alpha", type=float, default=0.6,
                        help="PER alpha parameter")
    parser.add_argument("--per-beta", type=float, default=0.4,
                        help="PER beta parameter")


def _add_network_arguments(parser):
    """Add neural network architecture arguments."""
    parser.add_argument("--layer-size", type=int, default=512,
                        help="Hidden layer size")
    parser.add_argument("--layer-size-bottleneck", type=int, default=50,
                        help="Encoder bottleneck size")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of hidden layers")


def _add_logging_arguments(parser):
    """Add logging and evaluation arguments."""
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Log progress every N episodes")
    parser.add_argument("--detailed-log-interval", type=int, default=50,
                        help="Detailed log every N episodes")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--metrics-save-interval", type=int, default=100,
                        help="Save metrics every N episodes")


def _add_checkpoint_arguments(parser):
    """Add checkpoint-related arguments."""
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save checkpoints every N episodes")
    parser.add_argument("--load-checkpoints", type=str, default=None,
                        help="Path to checkpoint file to resume from")
    parser.add_argument("--load-metrics", type=str, default=None,
                        help="Path to metrics file to resume from")