from types import SimpleNamespace


class GQNConfig:
    """Simple GQN config that extends DecQN config minimally."""

    @staticmethod
    def from_decqn_config(decqn_config):
        """Create GQN config from existing DecQN config."""
        # Copy all DecQN settings
        config = SimpleNamespace()
        for key, value in vars(decqn_config).items():
            setattr(config, key, value)

        # Add only GQN-specific parameters
        config.max_bins = 9  # Maximum resolution as per paper
        config.growing_schedule = "adaptive"  # Only adaptive schedule

        # Ensure decouple is True for GQN (as per paper)
        config.decouple = True

        return config

    @staticmethod
    def get_default_gqn_config(args):
        """Get default GQN config using DecQN as base."""
        # Import DecQN config creator
        from src.deqn.config import create_config_from_args

        # Create DecQN config first
        decqn_config = create_config_from_args(args)

        # Convert to GQN
        gqn_config = GQNConfig.from_decqn_config(decqn_config)

        # Override max_bins from args if provided
        if hasattr(args, "max_bins"):
            gqn_config.max_bins = args.max_bins

        return gqn_config
