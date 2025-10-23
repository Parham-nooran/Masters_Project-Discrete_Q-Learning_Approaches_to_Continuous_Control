def create_bangbang_config(args):
    """Create config.py for Bang-Bang agent."""
    from types import SimpleNamespace

    config = SimpleNamespace()
    for key, value in vars(args).items():
        setattr(config, key.replace("-", "_"), value)

    config.use_pixels = False
    config.layer_size_network = [512, 512]
    config.layer_size_bottleneck = 100
    config.num_pixels = 84
    config.min_replay_size = 1000
    config.max_replay_size = 500000
    config.batch_size = 128
    config.learning_rate = 3e-4
    config.discount = 0.99
    config.priority_exponent = 0.6
    config.importance_sampling_exponent = 0.4
    config.adder_n_step = 1
    config.clip_gradients = True
    config.clip_gradients_norm = 40.0

    return config
