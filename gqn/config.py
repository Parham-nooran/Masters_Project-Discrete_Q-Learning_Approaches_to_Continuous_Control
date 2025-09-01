"""
Growing Q-Networks configuration file.
This implements the hyperparameters as specified in the paper (Table 1 in Appendix).
"""

from types import SimpleNamespace


class GQNConfig:
    """Growing Q-Networks configuration based on the paper specifications."""

    @staticmethod
    def get_default_config(args):
        """Get default configuration matching paper specifications."""
        config = SimpleNamespace()

        for key, value in vars(args).items():
            setattr(config, key.replace("-", "_"), value)

        # ===== Core GQN Parameters =====
        config.algorithm = "gqn"
        config.decouple = True  # Always True for GQN
        config.max_bins = 9  # Maximum resolution as used in paper experiments
        config.growing_schedule = "adaptive"  # "linear" or "adaptive"
        config.growth_sequence = [2, 3, 5, 9]  # As specified in paper experiments

        # ===== Environment Parameters =====
        config.task = "walker_walk"
        config.use_pixels = False  # Paper focuses on state-based control
        config.action_penalty = 0.1  # Action penalty coefficient (ca)

        # ===== Network Architecture =====
        # Paper Table 1 hyperparameters
        if 'myo' in getattr(args, 'task', '').lower():
            config.layer_size_network = [2048, 2048]  # Extended capacity for MyoSuite
        else:
            config.layer_size_network = [512, 512]  # Standard network as per Table 1
        config.layer_size_bottleneck = 100  # For vision encoder (if used)
        config.num_pixels = 84  # Image size for pixel observations
        config.use_residual = False  # Paper uses simple MLP

        # ===== Training Parameters (Paper Table 1) =====
        config.learning_rate = 1e-4  # Adam learning rate
        config.discount = 0.99  # Discount factor γ
        config.batch_size = 256  # Batch size
        config.target_update_period = 100  # Target network update frequency
        config.epsilon = 0.1  # Exploration epsilon

        # ===== Experience Replay (Paper Table 1) =====
        config.max_replay_size = 500000  # Replay buffer capacity
        config.min_replay_size = 1000  # Minimum buffer size before training
        config.importance_sampling_exponent = 0.2  # β for PER
        config.priority_exponent = 0.6  # α for PER
        config.adder_n_step = 3  # N-step returns

        # ===== Q-Learning Specific =====
        config.use_double_q = True  # Double Q-learning
        config.huber_loss_parameter = 1.0  # Huber loss parameter
        config.clip_gradients = True  # Gradient clipping
        config.clip_gradients_norm = 40.0  # Clip norm as specified in paper

        # ===== Training Control =====
        config.num_episodes = 1000
        config.samples_per_insert = 32.0  # Training frequency

        return config

    @staticmethod
    def get_walker_config(args):
        """Configuration optimized for Walker tasks (as used in paper)."""
        config = GQNConfig.get_default_config(args)
        config.task = "walker_walk"
        config.action_penalty = 0.1
        config.num_episodes = 1000
        return config

    @staticmethod
    def get_humanoid_config(args):
        """Configuration for Humanoid tasks (higher dimensional)."""
        config = GQNConfig.get_default_config(args)
        config.task = "humanoid_stand"
        config.action_penalty = 0.1
        config.num_episodes = 1500  # More episodes for complex task
        config.layer_size_network = [2048, 2048]  # Larger network for complex task
        config.discount = 0.95  # Lower discount as used in MyoSuite experiments
        return config

    @staticmethod
    def get_manipulation_config(args):
        """Configuration for manipulation tasks (velocity control)."""
        config = GQNConfig.get_default_config(args)
        config.growth_sequence = [9, 17, 33, 65]  # Paper uses higher resolution for manipulation
        config.max_bins = 65
        config.action_penalty = 0.5  # Higher penalty for smoother control
        config.num_episodes = 2000
        return config

    @staticmethod
    def get_linear_schedule_config(args):
        """Configuration using linear growing schedule."""
        config = GQNConfig.get_default_config(args)
        config.growing_schedule = "linear"
        return config

    @staticmethod
    def from_args(args):
        """Create configuration from command line arguments."""
        config = GQNConfig.get_default_config(args)

        # Override with command line arguments
        for key, value in vars(args).items():
            if hasattr(config, key.replace("-", "_")):
                setattr(config, key.replace("-", "_"), value)

        # Task-specific adjustments
        if "humanoid" in config.task.lower():
            humanoid_config = GQNConfig.get_humanoid_config(args)
            # Keep user overrides but apply humanoid defaults for non-specified params
            for key, value in vars(humanoid_config).items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(config, key, value)

        return config

