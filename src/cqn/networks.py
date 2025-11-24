import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from src.common.encoder import VisionEncoder, LayerNormMLP


class DuelingHead(nn.Module):
    """
    Dueling network architecture for Q-value estimation.
    Splits into value and advantage streams.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()

        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class CQNNetwork(nn.Module):
    """
    Coarse-to-fine Q-Network that processes observations and outputs Q-values
    for each level and action dimension independently.

    KEY DIFFERENCES FROM YOUR IMPLEMENTATION:
    1. Uses dueling architecture
    2. Uses distributional critic (C51)
    3. Shares most parameters across levels via shared trunk
    4. Takes level index and previous actions as input
    """

    def __init__(
            self,
            config,
            obs_shape: Tuple,
            action_dim: int,
            num_levels: int = 3,
            num_bins: int = 5,
            use_distributional: bool = True,
            num_atoms: int = 51,
            v_min: float = 0.0,
            v_max: float = 200.0,
    ):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.num_levels = num_levels
        self.num_bins = num_bins
        self.use_distributional = use_distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

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

        shared_trunk_dim = config.layer_size_bottleneck

        self.shared_trunk = nn.Sequential(
            nn.Linear(critic_input_dim, shared_trunk_dim),
            nn.LayerNorm(shared_trunk_dim),
            nn.SiLU(),
            nn.Linear(shared_trunk_dim, shared_trunk_dim),
            nn.LayerNorm(shared_trunk_dim),
            nn.SiLU()
        )

        output_dim = num_bins * num_atoms if use_distributional else num_bins

        self.critics = nn.ModuleList([
            DuelingHead(shared_trunk_dim, output_dim, config.layer_size_bottleneck // 2)
            for _ in range(action_dim)
        ])

        self.target_critics = nn.ModuleList([
            DuelingHead(shared_trunk_dim, output_dim, config.layer_size_bottleneck // 2)
            for _ in range(action_dim)
        ])

        self.critics2 = nn.ModuleList([
            DuelingHead(shared_trunk_dim, output_dim, config.layer_size_bottleneck // 2)
            for _ in range(action_dim)
        ])

        self.target_critics2 = nn.ModuleList([
            DuelingHead(shared_trunk_dim, output_dim, config.layer_size_bottleneck // 2)
            for _ in range(action_dim)
        ])

        if use_distributional:
            self.register_buffer(
                "support",
                torch.linspace(v_min, v_max, num_atoms)
            )

        self.update_target_networks(tau=1.0)

    def forward(
            self,
            obs: torch.Tensor,
            level: int,
            prev_action: Optional[torch.Tensor] = None,
            use_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            obs: Observations [batch_size, ...]
            level: Current level (0 to num_levels-1)
            prev_action: Previous level actions [batch_size, action_dim] (None for level 0)
            use_target: Whether to use target network

        Returns:
            Q-values [batch_size, action_dim, num_bins]
        """
        if self.encoder:
            features = self.encoder(obs)
        else:
            features = obs.flatten(1)

        batch_size = features.shape[0]

        level_tensor = torch.tensor([level] * batch_size, dtype=torch.long, device=obs.device)
        level_emb = self.level_embedding(level_tensor)

        if prev_action is not None:
            prev_action = prev_action.float()
            prev_action_emb = self.prev_action_embedding(prev_action)
        else:
            prev_action_emb = torch.zeros(
                batch_size, self.prev_action_embedding.out_features,
                dtype=torch.float32, device=obs.device
            )

        combined_features = torch.cat([features, level_emb, prev_action_emb], dim=-1)

        shared_repr = self.shared_trunk(combined_features)

        q1_values = []
        q2_values = []
        critics1_to_use = self.target_critics if use_target else self.critics
        critics2_to_use = self.target_critics2 if use_target else self.critics2

        for dim in range(self.action_dim):
            q1_dim = critics1_to_use[dim](shared_repr)
            q2_dim = critics2_to_use[dim](shared_repr)

            if self.use_distributional:
                q1_dim = q1_dim.view(batch_size, self.num_bins, self.num_atoms)
                q2_dim = q2_dim.view(batch_size, self.num_bins, self.num_atoms)

                q1_probs = F.softmax(q1_dim, dim=-1)
                q2_probs = F.softmax(q2_dim, dim=-1)

                q1_dim = (q1_probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                q2_dim = (q2_probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

            q1_values.append(q1_dim)
            q2_values.append(q2_dim)

        q1 = torch.stack(q1_values, dim=1)
        q2 = torch.stack(q2_values, dim=1)

        return q1, q2

    def update_target_networks(self, tau: float = 0.005):
        """
        Update target networks with soft updates (Polyak averaging).
        """
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(
                    critic.parameters(), target_critic.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

        for critic, target_critic in zip(self.critics2, self.target_critics2):
            for param, target_param in zip(
                    critic.parameters(), target_critic.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )