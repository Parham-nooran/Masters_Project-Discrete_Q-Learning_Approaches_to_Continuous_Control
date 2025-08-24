import argparse
from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(description='Train DecQN Agent')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--task', type=str, default='walker_walk',
                        help='Environment task')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--num-steps', type=int, default=1000000,
                        help='Number of steps to train')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--mock-episode-length', type=int, default=100,
                        help='Mock episode length for training')
    parser.add_argument('--algorithm', type=str, default='decqnvis',
                        help='Algorithm to use')
    parser.add_argument('--num-bins', type=int, default=2,
                        help='Number of bins for discretization')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon for exploration')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--target-update-period', type=int, default=100,
                        help='Target network update period')
    parser.add_argument('--min-replay-size', type=int, default=1000,
                        help='Minimum replay buffer size')
    parser.add_argument('--max-replay-size', type=int, default=500000,
                        help='Maximum replay buffer size')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='Log progress every N episodes')
    parser.add_argument('--detailed-log-interval', type=int, default=25,
                        help='Detailed log every N episodes')
    return parser.parse_args()


def create_config_from_args(args):
    """Create config object from parsed arguments."""
    config = SimpleNamespace()

    # Copy all args to config
    for key, value in vars(args).items():
        setattr(config, key.replace('-', '_'), value)

    # Add derived properties and missing defaults
    config.decouple = 'decqn' in config.algorithm
    config.use_pixels = False
    config.use_double_q = True
    config.layer_size_network = [1024, 1024] if config.use_pixels else [512, 512]
    config.layer_size_bottleneck = 100
    config.num_pixels = 84
    config.pad_size = 4
    config.samples_per_insert = 32.0
    config.importance_sampling_exponent = 0.2
    config.priority_exponent = 0.6
    config.use_residual = False
    config.adder_n_step = 3
    config.huber_loss_parameter = 1.0
    config.clip_gradients = True
    config.clip_gradients_norm = 40.0

    return config

