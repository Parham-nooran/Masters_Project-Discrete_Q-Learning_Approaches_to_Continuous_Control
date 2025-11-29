import torch
import torch.nn as nn


class DecoupledQNetwork(nn.Module):
    """Decoupled Q-Network with per-dimension Q-values."""

    def __init__(self, obs_dim, action_dim, num_bins, layer_size=512, num_layers=2):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_bins = num_bins

        layers = []
        input_size = obs_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_size, layer_size),
                nn.LayerNorm(layer_size),
                nn.ELU()
            ])
            input_size = layer_size

        self.trunk = nn.Sequential(*layers)

        self.q_heads = nn.ModuleList([
            nn.Linear(layer_size, num_bins) for _ in range(action_dim)
        ])

    def forward(self, obs):
        """Forward pass returning per-dimension Q-values."""
        features = self.trunk(obs)

        q_values = []
        for head in self.q_heads:
            q_values.append(head(features))

        return torch.stack(q_values, dim=1)


class DualDecoupledQNetwork(nn.Module):
    """Dual Q-Network for double Q-learning."""

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