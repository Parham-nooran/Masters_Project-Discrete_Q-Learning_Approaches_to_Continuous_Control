from types import SimpleNamespace


def create_config_from_args(args):
    """Create config.py object from parsed arguments."""
    config = SimpleNamespace()
    for key, value in vars(args).items():
        setattr(config, key.replace("-", "_"), value)
    config.decouple = True
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
