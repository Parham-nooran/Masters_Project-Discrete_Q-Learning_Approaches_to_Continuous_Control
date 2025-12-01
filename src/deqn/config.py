from types import SimpleNamespace


def _normalize_argument_name(key):
    """Convert hyphenated argument names to underscored."""
    return key.replace("-", "_")


def _copy_arguments_to_config(config, args):
    """Copy all arguments to config object."""
    for key, value in vars(args).items():
        normalized_key = _normalize_argument_name(key)
        setattr(config, normalized_key, value)


def _set_core_parameters(config):
    """Set core algorithm parameters."""
    config.decouple = True
    config.use_pixels = False
    config.use_double_q = True


def _set_network_architecture(config):
    """Set network architecture parameters."""
    if config.use_pixels:
        config.layer_size_network = [1024, 1024]
    else:
        config.layer_size_network = [512, 512]

    config.layer_size_bottleneck = 100
    config.num_pixels = 84
    config.pad_size = 4


def _set_replay_buffer_parameters(config):
    """Set replay buffer parameters."""
    config.samples_per_insert = 32.0
    config.importance_sampling_exponent = 0.2
    config.priority_exponent = 0.6


def _set_training_parameters(config):
    """Set training-related parameters."""
    config.use_residual = False
    config.adder_n_step = 3
    config.huber_loss_parameter = 1.0


def _set_gradient_parameters(config):
    """Set gradient clipping parameters."""
    config.clip_gradients = True
    config.clip_gradients_norm = 40.0


def create_config_from_args(args):
    """Create config object from parsed arguments."""
    config = SimpleNamespace()

    _copy_arguments_to_config(config, args)
    _set_core_parameters(config)
    _set_network_architecture(config)
    _set_replay_buffer_parameters(config)
    _set_training_parameters(config)
    _set_gradient_parameters(config)

    return config