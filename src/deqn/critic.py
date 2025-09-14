import torch.nn as nn
from src.deqn.networks import LayerNormAndResidualMLP, LayerNormMLP


class CriticDQN(nn.Module):
    """Double Q-network critic for DecQN."""
    def __init__(self, config, input_size, action_spec):
        super().__init__()
        self.use_double_q = config.use_double_q
        self.decouple = config.decouple
        self.num_bins = config.num_bins

        if isinstance(action_spec, dict):
            if "low" in action_spec and "high" in action_spec:
                self.action_dims = len(action_spec["low"])
        elif hasattr(action_spec, "shape"):
            self.action_dims = action_spec.shape[0] if len(action_spec.shape) > 0 else 1
        elif hasattr(action_spec, "__len__"):
            self.action_dims = len(action_spec)
        else:
            self.action_dims = 1

        if config.decouple:
            self.output_dim = config.num_bins * self.action_dims
        else:
            self.output_dim = config.num_bins**self.action_dims

        if config.use_residual and not config.use_pixels:
            self.q1_network = nn.Sequential(
                nn.Flatten(),
                LayerNormAndResidualMLP(config.layer_size_network[0], num_blocks=1),
                nn.ELU(),
                nn.Linear(config.layer_size_network[0], self.output_dim),
            )
            self.q2_network = nn.Sequential(
                nn.Flatten(),
                LayerNormAndResidualMLP(config.layer_size_network[0], num_blocks=1),
                nn.ELU(),
                nn.Linear(config.layer_size_network[0], self.output_dim),
            )
        else:
            if config.use_pixels:
                sizes = [input_size] + config.layer_size_network + [self.output_dim]
                self.q1_network = LayerNormMLP(sizes)
                self.q2_network = LayerNormMLP(sizes)
            else:
                sizes = [input_size] + config.layer_size_network + [self.output_dim]
                self.q1_network = nn.Sequential(nn.Flatten(), LayerNormMLP(sizes))
                self.q2_network = nn.Sequential(nn.Flatten(), LayerNormMLP(sizes))

    def forward(self, x):
        q1 = self.q1_network(x)
        q2 = self.q2_network(x) if self.use_double_q else q1
        if self.decouple:
            q1 = q1.view(q1.shape[0], self.action_dims, self.num_bins)
            q2 = (
                q2.view(q2.shape[0], self.action_dims, self.num_bins)
                if self.use_double_q
                else q1
            )
        return q1, q2
