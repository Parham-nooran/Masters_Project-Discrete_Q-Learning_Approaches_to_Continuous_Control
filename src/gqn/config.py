from types import SimpleNamespace
from src.deqn.config import create_config_from_args


class GQNConfig:
    """Simple GQN config that extends DecQN config minimally."""

    @staticmethod
    def from_decqn_config(decqn_config):
        """Create GQN config from existing DecQN config."""
        config = SimpleNamespace()
        for key, value in vars(decqn_config).items():
            setattr(config, key, value)
        config.max_bins = 9
        config.growing_schedule = "adaptive"
        config.decouple = True
        return config

    @staticmethod
    def get_default_gqn_config(args):
        """Get default GQN config using DecQN as base."""
        decqn_config = create_config_from_args(args)
        gqn_config = GQNConfig.from_decqn_config(decqn_config)
        if hasattr(args, "max_bins"):
            gqn_config.max_bins = args.max_bins
        return gqn_config
