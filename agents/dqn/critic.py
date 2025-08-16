import torch.nn as nn
from networks import LayerNormAndResidualMLP, LayerNormMLP

class CriticDQN(nn.Module):
    """Double Q-network critic for DecQN."""

    def __init__(self, config, input_size, action_spec):
        super().__init__()
        self.use_double_q = config.use_double_q
        self.decouple = config.decouple
        self.num_bins = config.num_bins

        # Calculate output dimensions
        if config.decouple:
            self.output_dim = config.num_bins * len(action_spec.shape) if hasattr(action_spec,
                                                                                  'shape') else (config.num_bins *
                                                                                                 action_spec)
            self.action_dims = len(action_spec.shape) if hasattr(action_spec, 'shape') else action_spec
        else:
            self.output_dim = config.num_bins ** (
                len(action_spec.shape) if hasattr(action_spec, 'shape') else action_spec)

        # Build networks
        if config.use_residual and not config.use_pixels:
            self.q1_network = nn.Sequential(
                nn.Flatten(),
                LayerNormAndResidualMLP(config.layer_size_network[0], num_blocks=1),
                nn.ELU(),
                nn.Linear(config.layer_size_network[0], self.output_dim)
            )
            self.q2_network = nn.Sequential(
                nn.Flatten(),
                LayerNormAndResidualMLP(config.layer_size_network[0], num_blocks=1),
                nn.ELU(),
                nn.Linear(config.layer_size_network[0], self.output_dim)
            )
        else:
            if config.use_pixels:
                # For pixels, input_size is from encoder output
                sizes = [input_size] + config.layer_size_network + [self.output_dim]
                self.q1_network = LayerNormMLP(sizes)
                self.q2_network = LayerNormMLP(sizes)
            else:
                # For state inputs, flatten first
                sizes = [input_size] + config.layer_size_network + [self.output_dim]
                self.q1_network = nn.Sequential(nn.Flatten(), LayerNormMLP(sizes))
                self.q2_network = nn.Sequential(nn.Flatten(), LayerNormMLP(sizes))

    def forward(self, x):
        q1 = self.q1_network(x)
        q2 = self.q2_network(x) if self.use_double_q else q1

        if self.decouple:
            q1 = q1.view(q1.shape[0], self.action_dims, self.num_bins)
            q2 = q2.view(q2.shape[0], self.action_dims, self.num_bins) if self.use_double_q else q1

        return q1, q2
