"""
Neural network architectures for CQN agent.

Implements the distributional critic with dueling architecture and 
hierarchical action discretization following the CQN paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RandomShiftsAug(nn.Module):
    """Random shift augmentation for image observations."""

    def __init__(self, pad: int = 4):
        """
        Initialize random shift augmentation.

        Args:
            pad: Padding size for shifts.
        """
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random shifts to images.

        Args:
            x: Image tensor [B, C, H, W].

        Returns:
            Augmented image tensor.
        """
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


class ImgChLayerNorm(nn.Module):
    """Layer normalization for image channels."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        """
        Initialize channel-wise layer normalization.

        Args:
            num_channels: Number of channels.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise layer normalization.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Normalized tensor.
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Encoder(nn.Module):
    """Convolutional encoder for visual observations."""

    def __init__(self, obs_shape: Tuple):
        """
        Initialize encoder.

        Args:
            obs_shape: Shape of observations (C, H, W).
        """
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=True),
        )

        self.apply(self._weight_init)

    def _weight_init(self, m):
        """Initialize weights orthogonally."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(m.weight.data, gain)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations.

        Args:
            obs: Observation tensor [B, C, H, W].

        Returns:
            Encoded features [B, repr_dim].
        """
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class C2FCriticNetwork(nn.Module):
    """
    Coarse-to-fine critic network with dueling architecture.

    Outputs distributional Q-values for each action dimension and bin.
    """

    def __init__(
            self,
            repr_dim: int,
            action_shape: Tuple,
            feature_dim: int,
            hidden_dim: int,
            levels: int,
            bins: int,
            atoms: int,
    ):
        """
        Initialize critic network.

        Args:
            repr_dim: Dimension of encoded observations.
            action_shape: Shape of action space.
            feature_dim: Feature dimension.
            hidden_dim: Hidden layer dimension.
            levels: Number of hierarchy levels.
            bins: Number of bins per level.
            atoms: Number of atoms for distributional RL.
        """
        super().__init__()
        self._levels = levels
        self._actor_dim = action_shape[0]
        self._bins = bins

        self.adv_trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_net = nn.Sequential(
            nn.Linear(feature_dim + self._actor_dim + levels, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.adv_head = nn.Linear(hidden_dim, self._actor_dim * bins * atoms)
        self.adv_output_shape = (self._actor_dim, bins, atoms)

        self.value_trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim + self._actor_dim + levels, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.value_head = nn.Linear(hidden_dim, self._actor_dim * 1 * atoms)
        self.value_output_shape = (self._actor_dim, 1, atoms)

        self.apply(self._weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def _weight_init(self, m):
        """Initialize weights orthogonally."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    def forward(
            self, level: int, obs: torch.Tensor, prev_action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            level: Current hierarchy level.
            obs: Encoded observation features.
            prev_action: Actions from previous level.

        Returns:
            Q-value logits [B, action_dim, bins, atoms].
        """
        level_id = (
            torch.eye(self._levels, device=obs.device, dtype=obs.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(obs.shape[0], 0)
        )

        value_h = self.value_trunk(obs)
        value_x = torch.cat([value_h, prev_action, level_id], -1)
        values = self.value_head(self.value_net(value_x)).view(
            -1, *self.value_output_shape
        )

        adv_h = self.adv_trunk(obs)
        adv_x = torch.cat([adv_h, prev_action, level_id], -1)
        advs = self.adv_head(self.adv_net(adv_x)).view(-1, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits


class C2FCritic(nn.Module):
    """
    Coarse-to-fine distributional critic.

    Uses C51 to model value distributions across hierarchical action levels.
    """

    def __init__(
            self,
            action_shape: Tuple,
            repr_dim: int,
            feature_dim: int,
            hidden_dim: int,
            levels: int,
            bins: int,
            atoms: int,
            v_min: float,
            v_max: float,
    ):
        """
        Initialize critic.

        Args:
            action_shape: Shape of action space.
            repr_dim: Dimension of encoded observations.
            feature_dim: Feature dimension.
            hidden_dim: Hidden layer dimension.
            levels: Number of hierarchy levels.
            bins: Number of bins per level.
            atoms: Number of atoms for distributional RL.
            v_min: Minimum value for distribution support.
            v_max: Maximum value for distribution support.
        """
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        actor_dim = action_shape[0]

        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )
        self.support = nn.Parameter(
            torch.linspace(v_min, v_max, atoms), requires_grad=False
        )
        self.delta_z = (v_max - v_min) / (atoms - 1)

        self.network = C2FCriticNetwork(
            repr_dim, action_shape, feature_dim, hidden_dim, levels, bins, atoms
        )

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Select action using coarse-to-fine discretization.

        Args:
            obs: Encoded observation features.

        Returns:
            Continuous action and metrics dictionary.
        """
        from src.cqn.cqn_utils import random_action_if_within_delta, zoom_in

        metrics = {}
        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        for level in range(self.levels):
            q_logits = self.network(level, obs, (low + high) / 2)
            q_probs = F.softmax(q_logits, 3)
            qs = (q_probs * self.support.expand_as(q_probs).detach()).sum(3)

            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]

            low, high = zoom_in(low, high, argmax_q, self.bins)

            qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[..., 0]
            metrics[f"critic_target_q_level{level}"] = qs_a.mean().item()

        continuous_action = (high + low) / 2.0
        return continuous_action, metrics

    def forward(
            self, obs: torch.Tensor, continuous_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute value distributions for given observations and actions.

        Args:
            obs: Encoded observation features [B, repr_dim].
            continuous_action: Continuous actions [B, action_dim].

        Returns:
            Tuple of (q_probs, q_probs_a, log_q_probs, log_q_probs_a) where:
                q_probs: [B, L, D, bins, atoms] - full distribution
                q_probs_a: [B, L, D, atoms] - distribution at selected actions
                log_q_probs: [B, L, D, bins, atoms] - log probabilities
                log_q_probs_a: [B, L, D, atoms] - log probs at selected actions
        """
        from src.cqn.cqn_utils import encode_action, zoom_in

        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        q_probs_per_level = []
        q_probs_a_per_level = []
        log_q_probs_per_level = []
        log_q_probs_a_per_level = []

        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        for level in range(self.levels):
            q_logits = self.network(level, obs, (low + high) / 2)
            argmax_q = discrete_action[..., level, :].long()

            q_probs = F.softmax(q_logits, 3)
            q_probs_a = torch.gather(
                q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1).unsqueeze(-1).repeat_interleave(self.atoms, -1),
            )
            q_probs_a = q_probs_a[..., 0, :]

            log_q_probs = F.log_softmax(q_logits, 3)
            log_q_probs_a = torch.gather(
                log_q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1).unsqueeze(-1).repeat_interleave(self.atoms, -1),
            )
            log_q_probs_a = log_q_probs_a[..., 0, :]

            q_probs_per_level.append(q_probs)
            q_probs_a_per_level.append(q_probs_a)
            log_q_probs_per_level.append(log_q_probs)
            log_q_probs_a_per_level.append(log_q_probs_a)

            low, high = zoom_in(low, high, argmax_q, self.bins)

        q_probs = torch.stack(q_probs_per_level, -4)
        q_probs_a = torch.stack(q_probs_a_per_level, -3)
        log_q_probs = torch.stack(log_q_probs_per_level, -4)
        log_q_probs_a = torch.stack(log_q_probs_a_per_level, -3)

        return q_probs, q_probs_a, log_q_probs, log_q_probs_a

    def compute_target_q_dist(
            self,
            next_obs: torch.Tensor,
            next_continuous_action: torch.Tensor,
            reward: torch.Tensor,
            discount: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target distribution for distributional critic.

        Uses categorical projection from C51 algorithm.

        Args:
            next_obs: Next observation features [B, repr_dim].
            next_continuous_action: Next actions [B, action_dim].
            reward: Rewards [B, 1].
            discount: Discount factors [B, 1].

        Returns:
            Target distribution [B, L, D, atoms].
        """
        next_q_probs_a = self.forward(next_obs, next_continuous_action)[1]

        shape = next_q_probs_a.shape
        next_q_probs_a = next_q_probs_a.view(-1, self.atoms)
        batch_size = next_q_probs_a.shape[0]

        Tz = reward + discount * self.support.unsqueeze(0).detach()
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)

        b = (Tz - self.v_min) / self.delta_z
        lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)

        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.atoms - 1)) * (lower == upper)] += 1

        multiplier = batch_size // lower.shape[0]
        b = torch.repeat_interleave(b, multiplier, 0)
        lower = torch.repeat_interleave(lower, multiplier, 0)
        upper = torch.repeat_interleave(upper, multiplier, 0)

        m = torch.zeros_like(next_q_probs_a)
        offset = (
            torch.linspace(
                0,
                ((batch_size - 1) * self.atoms),
                batch_size,
                device=lower.device,
                dtype=lower.dtype,
            )
            .unsqueeze(1)
            .expand(batch_size, self.atoms)
        )

        m.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_q_probs_a * (upper.float() - b)).view(-1),
        )
        m.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_q_probs_a * (b - lower.float())).view(-1),
        )

        m = m.view(*shape)
        return m


class CQNNetwork(nn.Module):
    """
    Complete CQN network with encoder and critic.

    Handles both pixel and state-based observations.
    """

    def __init__(
            self,
            config,
            obs_shape: Tuple,
            action_dim: int,
            num_levels: int,
            num_bins: int,
            num_atoms: int,
            v_min: float,
            v_max: float,
    ):
        """
        Initialize CQN network.

        Args:
            config: Configuration object.
            obs_shape: Shape of observations.
            action_dim: Dimension of action space.
            num_levels: Number of hierarchy levels.
            num_bins: Number of bins per level.
            num_atoms: Number of atoms for distributional RL.
            v_min: Minimum value for distribution support.
            v_max: Maximum value for distribution support.
        """
        super().__init__()

        self.use_pixels = len(obs_shape) == 3

        if self.use_pixels:
            self.encoder = Encoder(obs_shape)
            repr_dim = self.encoder.repr_dim
            self.aug = RandomShiftsAug(pad=4)
        else:
            self.encoder = None
            repr_dim = obs_shape[0]
            self.aug = None

        self.critic = C2FCritic(
            action_shape=(action_dim,),
            repr_dim=repr_dim,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            levels=num_levels,
            bins=num_bins,
            atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
        )