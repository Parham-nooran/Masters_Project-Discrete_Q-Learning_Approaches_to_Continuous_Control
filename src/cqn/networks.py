from typing import Tuple, Optional

import torch
import torch.nn as nn

from src.common.encoder import VisionEncoder


class DuelingHead(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()

        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, output_dim)

        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.advantage_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        value = self.value_head(self.value_stream(x))
        advantage = self.advantage_head(self.advantage_stream(x))

        q_values = value + (advantage - advantage.mean(dim=-2, keepdim=True))
        return q_values


class CQNNetwork(nn.Module):

    def __init__(
            self,
            config,
            obs_shape: Tuple,
            action_dim: int,
            num_levels: int = 3,
            num_bins: int = 5,
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
            nn.Linear(critic_input_dim, shared_trunk_dim, bias=False),
            nn.LayerNorm(shared_trunk_dim),
            nn.SiLU(),
            nn.Linear(shared_trunk_dim, shared_trunk_dim, bias=False),
            nn.LayerNorm(shared_trunk_dim),
            nn.SiLU()
        )

        output_dim = num_bins * num_atoms

        self.critics = nn.ModuleList([
            DuelingHead(shared_trunk_dim, output_dim, config.layer_size_bottleneck // 2)
            for _ in range(action_dim)
        ])

        self.target_critics = nn.ModuleList([
            DuelingHead(shared_trunk_dim, output_dim, config.layer_size_bottleneck // 2)
            for _ in range(action_dim)
        ])

        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms)
        )

        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self._initialize_weights()
        self.update_target_networks(tau=1.0)

    def _initialize_weights(self):
        for critic in self.critics:
            critic.value_head.weight.data.fill_(0.0)
            critic.value_head.bias.data.fill_(0.0)
            critic.advantage_head.weight.data.fill_(0.0)
            critic.advantage_head.bias.data.fill_(0.0)

    def forward(
            self,
            obs: torch.Tensor,
            level: int,
            prev_action: Optional[torch.Tensor] = None,
            use_target: bool = False,
    ) -> torch.Tensor:
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

        q_values = []
        critics_to_use = self.target_critics if use_target else self.critics

        for dim in range(self.action_dim):
            q_dim = critics_to_use[dim](shared_repr)
            q_dim = q_dim.view(batch_size, self.num_bins, self.num_atoms)
            q_values.append(q_dim)

        q = torch.stack(q_values, dim=1)

        return q

    def update_target_networks(self, tau: float = 0.005):
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(
                    critic.parameters(), target_critic.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
