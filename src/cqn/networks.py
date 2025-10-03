import torch.nn as nn
import torch
from typing import Tuple, Optional
from src.common.encoder import VisionEncoder, LayerNormMLP


class CQNNetwork(nn.Module):
    """
    Coarse-to-fine Q-Network that processes observations and outputs Q-values
    for each level and action dimension independently.
    """

    def __init__(
        self,
        config,
        obs_shape: Tuple,
        action_dim: int,
        num_levels: int = 3,
        num_bins: int = 5,
    ):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.num_levels = num_levels
        self.num_bins = num_bins
        self.use_pixels = len(obs_shape) > 1
        if self.use_pixels:
            self.encoder = VisionEncoder(config, shape=obs_shape[-1])
            feature_dim = config.layer_size_bottleneck
        else:
            feature_dim = obs_shape[0]
            self.encoder = None
        self.level_embedding = nn.Embedding(
            num_levels, config.layer_size_bottleneck // 4
        )
        self.prev_action_embedding = nn.Linear(
            action_dim, config.layer_size_bottleneck // 4
        )
        critic_input_dim = (
            feature_dim
            + config.layer_size_bottleneck // 4
            + config.layer_size_bottleneck // 4
        )
        self.critics = nn.ModuleList(
            [
                LayerNormMLP(
                    [
                        critic_input_dim,
                        config.layer_size_bottleneck,
                        config.layer_size_bottleneck // 2,
                        num_bins,
                    ]
                )
                for _ in range(action_dim)
            ]
        )

        self.target_critics = nn.ModuleList(
            [
                LayerNormMLP(
                    [
                        critic_input_dim,
                        config.layer_size_bottleneck,
                        config.layer_size_bottleneck // 2,
                        num_bins,
                    ]
                )
                for _ in range(action_dim)
            ]
        )

        self.update_target_networks(tau=1.0)

    def forward(
        self, obs: torch.Tensor, level: int, prev_action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            obs: Observations [batch_size, ...]
            level: Current level (0 to num_levels-1)
            prev_action: Previous level actions [batch_size, action_dim] (None for level 0)

        Returns:
            Tuple of Q-values from two critic networks [batch_size, action_dim, num_bins]
        """
        if self.encoder:
            features = self.encoder(obs)
        else:
            features = obs.flatten(1)

        batch_size = features.shape[0]

        level_tensor = torch.full(
            (batch_size,), level, dtype=torch.long, device=obs.device
        )
        level_emb = self.level_embedding(level_tensor)

        if prev_action is not None:
            prev_action_emb = self.prev_action_embedding(prev_action)
        else:
            prev_action_emb = torch.zeros(
                batch_size, self.prev_action_embedding.out_features, device=obs.device
            )

        combined_features = torch.cat([features, level_emb, prev_action_emb], dim=-1)

        q1_values = []
        q2_values = []

        for dim in range(self.action_dim):
            q1_dim = self.critics[dim](combined_features)
            q2_dim = self.target_critics[dim](combined_features)
            q1_values.append(q1_dim)
            q2_values.append(q2_dim)

        q1 = torch.stack(q1_values, dim=1)
        q2 = torch.stack(q2_values, dim=1)

        return q1, q2

    def update_target_networks(self, tau: float = 0.005):
        """Update target networks with soft updates"""
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(
                critic.parameters(), target_critic.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
