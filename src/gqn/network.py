import torch
import torch.nn as nn


class DecoupledQNetwork(nn.Module):
    """Decoupled Q-Network with per-dimension Q-values."""

    def __init__(self, obs_dim, action_dim, num_bins, layer_size=512, num_layers=2):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_bins = num_bins

        self.trunk = self._build_trunk_network(obs_dim, layer_size, num_layers)
        self.q_heads = self._build_q_heads(action_dim, layer_size, num_bins)

    def _build_trunk_network(self, obs_dim, layer_size, num_layers):
        """Build trunk network for feature extraction."""
        layers = []
        input_size = obs_dim

        for _ in range(num_layers):
            layers.extend(self._create_layer_block(input_size, layer_size))
            input_size = layer_size

        return nn.Sequential(*layers)

    def _create_layer_block(self, input_size, output_size):
        """Create a single layer block with normalization and activation."""
        return [
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
            nn.ELU()
        ]

    def _build_q_heads(self, action_dim, layer_size, num_bins):
        """Build separate Q-value heads for each action dimension."""
        return nn.ModuleList([
            nn.Linear(layer_size, num_bins) for _ in range(action_dim)
        ])

    def forward(self, obs):
        """Forward pass returning per-dimension Q-values."""
        features = self.trunk(obs)
        q_values = self._compute_q_values_per_dimension(features)
        return torch.stack(q_values, dim=1)

    def _compute_q_values_per_dimension(self, features):
        """Compute Q-values for each action dimension."""
        return [head(features) for head in self.q_heads]


class DualDecoupledQNetwork(nn.Module):
    """Dual Q-Network for double Q-learning with decoupled action dimensions."""

    def __init__(self, obs_dim, action_dim, num_bins, layer_size=512, num_layers=2):
        super().__init__()

        self.q1 = DecoupledQNetwork(obs_dim, action_dim, num_bins, layer_size, num_layers)
        self.q2 = DecoupledQNetwork(obs_dim, action_dim, num_bins, layer_size, num_layers)

    def forward(self, obs):
        """Forward pass returning both Q-networks."""
        return self.q1(obs), self.q2(obs)

    def get_q1(self, obs):
        """Get Q1 network output."""
        return self.q1(obs)

    def get_q2(self, obs):
        """Get Q2 network output."""
        return self.q2(obs)