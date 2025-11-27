from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import src.cqn.utils as utils

class C2FCriticNetwork(nn.Module):
    def __init__(
            self,
            repr_dim: int,
            low_dim: int,
            action_shape: Tuple,
            feature_dim: int,
            hidden_dim: int,
            levels: int,
            bins: int,
            atoms: int,
    ):
        super().__init__()
        self._levels = levels
        self._actor_dim = action_shape[0]
        self._bins = bins

        self.adv_rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._actor_dim + levels, hidden_dim, bias=False
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.adv_head = nn.Linear(hidden_dim, self._actor_dim * bins * atoms)
        self.adv_output_shape = (self._actor_dim, bins, atoms)

        # Value stream in Dueling network
        self.value_rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._actor_dim + levels, hidden_dim, bias=False
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(hidden_dim, self._actor_dim * 1 * atoms)
        self.value_output_shape = (self._actor_dim, 1, atoms)

        self.apply(utils.weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def forward(
            self, level: int, rgb_obs: torch.Tensor, low_dim_obs, prev_action: torch.Tensor
    ):
        """
        Inputs:
        - level: level index
        - obs: features from visual encoder
        - low_dim_obs: low-dimensional observations
        - prev_action: actions from previous level

        Outputs:
        - q_logits: (batch_size, action_dim, bins, atoms)
        """
        level_id = (
            torch.eye(self._levels, device=rgb_obs.device, dtype=rgb_obs.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(rgb_obs.shape[0], 0)
        )

        value_h = torch.cat(
            [self.value_rgb_encoder(rgb_obs), self.value_low_dim_encoder(low_dim_obs)],
            -1,
        )
        value_x = torch.cat([value_h, prev_action, level_id], -1)
        values = self.value_head(self.value_net(value_x)).view(
            -1, *self.value_output_shape
        )

        adv_h = torch.cat(
            [self.adv_rgb_encoder(rgb_obs), self.adv_low_dim_encoder(low_dim_obs)], -1
        )
        adv_x = torch.cat([adv_h, prev_action, level_id], -1)
        advs = self.adv_head(self.adv_net(adv_x)).view(-1, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits


class ImgChLayerNorm(nn.Module):
    def __init__(self, num_channels, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
