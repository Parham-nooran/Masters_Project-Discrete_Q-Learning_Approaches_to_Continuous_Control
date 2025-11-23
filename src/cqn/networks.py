import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from src.common.encoder import VisionEncoder, LayerNormMLP


class DuelingC51Head(nn.Module):
    def __init__(self, input_dim: int, num_bins: int, num_atoms: int = 51):
        super().__init__()
        self.num_bins = num_bins
        self.num_atoms = num_atoms

        hidden_dim = input_dim // 2

        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atoms, bias=False)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins * num_atoms, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        value = value.view(batch_size, 1, self.num_atoms)
        advantages = advantages.view(batch_size, self.num_bins, self.num_atoms)

        advantages_mean = advantages.mean(dim=1, keepdim=True)
        q_atoms = value + (advantages - advantages_mean)

        logits = q_atoms
        probs = F.softmax(logits, dim=-1)

        return probs


class CQNNetwork(nn.Module):
    def __init__(
            self,
            config,
            obs_shape: Tuple,
            action_dim: int,
            num_levels: int = 3,
            num_bins: int = 5,
            num_atoms: int = 51,
            v_min: float = -1.0,
            v_max: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.num_levels = num_levels
        self.num_bins = num_bins
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

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
            action_dim, config.layer_size_bottleneck // 4, bias=False
        )

        critic_input_dim = (
                feature_dim
                + config.layer_size_bottleneck // 4
                + config.layer_size_bottleneck // 4
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(critic_input_dim, config.layer_size_bottleneck, bias=False),
            nn.LayerNorm(config.layer_size_bottleneck),
            nn.SiLU(),
            nn.Linear(config.layer_size_bottleneck, config.layer_size_bottleneck // 2, bias=False),
            nn.LayerNorm(config.layer_size_bottleneck // 2),
            nn.SiLU(),
        )

        shared_out_dim = config.layer_size_bottleneck // 2

        self.critics = nn.ModuleList(
            [DuelingC51Head(shared_out_dim, num_bins, num_atoms) for _ in range(action_dim)]
        )

        self.target_critics = nn.ModuleList(
            [DuelingC51Head(shared_out_dim, num_bins, num_atoms) for _ in range(action_dim)]
        )

        self.critics2 = nn.ModuleList(
            [DuelingC51Head(shared_out_dim, num_bins, num_atoms) for _ in range(action_dim)]
        )

        self.target_critics2 = nn.ModuleList(
            [DuelingC51Head(shared_out_dim, num_bins, num_atoms) for _ in range(action_dim)]
        )

        self.update_target_networks(tau=1.0)

    def forward(
            self,
            obs: torch.Tensor,
            level: int,
            prev_action: Optional[torch.Tensor] = None,
            use_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        shared_out = self.shared_layers(combined_features)

        q1_dists = []
        q2_dists = []
        critics1_to_use = self.target_critics if use_target else self.critics
        critics2_to_use = self.target_critics2 if use_target else self.critics2

        for dim in range(self.action_dim):
            q1_dist = critics1_to_use[dim](shared_out)
            q2_dist = critics2_to_use[dim](shared_out)
            q1_dists.append(q1_dist)
            q2_dists.append(q2_dist)

        q1 = torch.stack(q1_dists, dim=1)
        q2 = torch.stack(q2_dists, dim=1)

        return q1, q2

    def get_q_values(self, q_dist: torch.Tensor) -> torch.Tensor:
        support = self.support.to(q_dist.device)
        return (q_dist * support.view(1, 1, -1)).sum(dim=-1)

    def update_target_networks(self, tau: float = 0.02):
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

    def forward_batched_levels(
            self,
            obs: torch.Tensor,
            level_indices: torch.Tensor,
            prev_action: Optional[torch.Tensor] = None,
            use_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder:
            features = self.encoder(obs)
        else:
            features = obs.flatten(1)

        batch_expanded = features.shape[0]
        level_emb = self.level_embedding(level_indices)

        if prev_action is not None:
            prev_action_emb = self.prev_action_embedding(prev_action)
        else:
            prev_action_emb = torch.zeros(
                batch_expanded, self.prev_action_embedding.out_features, device=obs.device
            )

        combined_features = torch.cat([features, level_emb, prev_action_emb], dim=-1)
        shared_out = self.shared_layers(combined_features)

        q1_dists = []
        q2_dists = []
        critics1_to_use = self.target_critics if use_target else self.critics
        critics2_to_use = self.target_critics2 if use_target else self.critics2

        for dim in range(self.action_dim):
            q1_dist = critics1_to_use[dim](shared_out)
            q2_dist = critics2_to_use[dim](shared_out)
            q1_dists.append(q1_dist)
            q2_dists.append(q2_dist)

        q1 = torch.stack(q1_dists, dim=1)
        q2 = torch.stack(q2_dists, dim=1)

        return q1, q2