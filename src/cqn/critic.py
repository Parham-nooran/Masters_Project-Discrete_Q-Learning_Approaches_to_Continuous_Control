import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cqn.networks import C2FCriticNetwork


def _repeat_bounds(action, bounds):
    """Repeat bounds to match action shape."""
    return bounds.repeat(*action.shape[:-1], 1)


class ActionEncoder:
    """Encodes and decodes continuous actions to/from discrete representations."""

    def __init__(self, initial_low, initial_high, levels, bins):
        self.initial_low = initial_low
        self.initial_high = initial_high
        self.levels = levels
        self.bins = bins

    def encode(self, continuous_action):
        """Encode continuous action to discrete action across levels."""
        low = _repeat_bounds(continuous_action, self.initial_low)
        high = _repeat_bounds(continuous_action, self.initial_high)

        discrete_indices = []

        for _ in range(self.levels):
            idx, low, high = self._encode_single_level(
                continuous_action, low, high
            )
            discrete_indices.append(idx)

        return torch.stack(discrete_indices, dim=-2)

    def decode(self, discrete_action):
        """Decode discrete action to continuous action."""
        low = _repeat_bounds(discrete_action[..., 0, :], self.initial_low)
        high = _repeat_bounds(discrete_action[..., 0, :], self.initial_high)

        for level in range(self.levels):
            continuous_action, low, high = self._decode_single_level(
                discrete_action[..., level, :], low, high
            )

        return (high + low) / 2.0

    def _encode_single_level(self, continuous_action, low, high):
        """Encode continuous action at a single level."""
        slice_range = (high - low) / self.bins
        idx = torch.floor((continuous_action - low) / slice_range)
        idx = torch.clip(idx, 0, self.bins - 1)

        new_action = low + slice_range * idx
        new_action = torch.clip(new_action, -1.0, 1.0)

        new_low = new_action
        new_high = new_action + slice_range
        new_low = torch.maximum(-torch.ones_like(new_low), new_low)
        new_high = torch.minimum(torch.ones_like(new_high), new_high)

        return idx, new_low, new_high

    def _decode_single_level(self, discrete_idx, low, high):
        """Decode discrete action at a single level."""
        slice_range = (high - low) / self.bins
        continuous_action = low + slice_range * discrete_idx

        new_low = continuous_action
        new_high = continuous_action + slice_range
        new_low = torch.maximum(-torch.ones_like(new_low), new_low)
        new_high = torch.minimum(torch.ones_like(new_high), new_high)

        return continuous_action, new_low, new_high


class IntervalZoomer:
    """Handles zooming into action intervals across levels."""

    def __init__(self, bins):
        self.bins = bins

    def zoom_in(self, low, high, bin_index):
        """Zoom into the selected bin interval."""
        slice_range = (high - low) / self.bins
        continuous_action = low + slice_range * bin_index

        new_low = continuous_action
        new_high = continuous_action + slice_range
        new_low = torch.maximum(-torch.ones_like(new_low), new_low)
        new_high = torch.minimum(torch.ones_like(new_high), new_high)

        return new_low, new_high


class RandomActionSelector:
    """Selects random actions when Q-values are within delta."""

    def __init__(self, delta=0.0001):
        self.delta = delta

    def select_or_argmax(self, q_values):
        """Select random action if Q-values are close, otherwise argmax."""
        q_diff = q_values.max(-1).values - q_values.min(-1).values
        random_mask = q_diff < self.delta

        if random_mask.sum() == 0:
            return None

        argmax_q = q_values.max(-1)[1]
        random_actions = torch.randint(
            0, q_values.size(-1), random_mask.shape
        ).to(q_values.device)

        return torch.where(random_mask, random_actions, argmax_q)


class QValueCalculator:
    """Calculates Q-values from probability distributions."""

    def __init__(self, support):
        self.support = support

    def calculate(self, q_probs):
        """Calculate Q-values from probability distribution."""
        expanded_support = self.support.expand_as(q_probs)
        return (q_probs * expanded_support.detach()).sum(dim=3)


def _expand_for_batch(b, lower, upper, batch_size, reward_batch_size):
    """Expand tensors to full batch size."""
    multiplier = batch_size // reward_batch_size
    b = torch.repeat_interleave(b, multiplier, dim=0)
    lower = torch.repeat_interleave(lower, multiplier, dim=0)
    upper = torch.repeat_interleave(upper, multiplier, dim=0)
    return b, lower, upper


def _add_lower_probabilities(m, probs, lower, upper, b, offset):
    """Add probability mass to lower bins."""
    m.view(-1).index_add_(
        0,
        (lower + offset).view(-1),
        (probs * (upper.float() - b)).view(-1),
    )


def _add_upper_probabilities(m, probs, lower, upper, b, offset):
    """Add probability mass to upper bins."""
    m.view(-1).index_add_(
        0,
        (upper + offset).view(-1),
        (probs * (b - lower.float())).view(-1),
    )


class DistributionalProjection:
    """Projects distributional Q-values for target computation."""

    def __init__(self, v_min, v_max, atoms):
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.delta_z = (v_max - v_min) / (atoms - 1)

    def project(self, next_q_probs, reward, discount, support):
        """Project next Q-value distribution onto current support."""
        shape = next_q_probs.shape
        next_q_probs_flat = next_q_probs.view(-1, self.atoms)
        batch_size = next_q_probs_flat.shape[0]

        projected_values = self._compute_projected_values(
            reward, discount, support
        )

        lower, upper = self._compute_projection_indices(projected_values)

        projected_values, lower, upper = _expand_for_batch(
            projected_values, lower, upper, batch_size, reward.shape[0]
        )

        projected_dist = self._distribute_probabilities(
            next_q_probs_flat, projected_values, lower, upper, batch_size
        )

        return projected_dist.view(*shape)

    def _compute_projected_values(self, reward, discount, support):
        """Compute projected values for distributional Bellman update."""
        Tz = reward + discount * support.unsqueeze(0).detach()
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        return b

    def _compute_projection_indices(self, b):
        """Compute lower and upper projection indices."""
        lower = b.floor().to(torch.int64)
        upper = b.ceil().to(torch.int64)

        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.atoms - 1)) * (lower == upper)] += 1

        return lower, upper

    def _distribute_probabilities(self, next_q_probs, b, lower, upper, batch_size):
        """Distribute probability mass to projection bins."""
        m = torch.zeros_like(next_q_probs)
        offset = self._compute_offset(batch_size, lower.device, lower.dtype)

        _add_lower_probabilities(m, next_q_probs, lower, upper, b, offset)
        _add_upper_probabilities(m, next_q_probs, lower, upper, b, offset)

        return m

    def _compute_offset(self, batch_size, device, dtype):
        """Compute offset for indexing."""
        return (
            torch.linspace(
                0,
                ((batch_size - 1) * self.atoms),
                batch_size,
                device=device,
                dtype=dtype,
            )
            .unsqueeze(1)
            .expand(batch_size, self.atoms)
        )


def _initialize_bounds(rgb_obs, bounds):
    """Initialize bounds for batch."""
    return bounds.repeat(rgb_obs.shape[0], 1).detach()


def _record_metrics(level, q_values, bin_index, metrics):
    """Record metrics for this level."""
    selected_q = torch.gather(
        q_values,
        dim=-1,
        index=bin_index.unsqueeze(-1)
    )[..., 0]
    metrics[f"critic_target_q_level{level}"] = selected_q.mean().item()


class HierarchicalActionSelector:
    """Selects actions hierarchically across coarse-to-fine levels."""

    def __init__(self, network, initial_low, initial_high, levels, bins, support):
        self.network = network
        self.initial_low = initial_low
        self.initial_high = initial_high
        self.levels = levels
        self.interval_zoomer = IntervalZoomer(bins)
        self.random_selector = RandomActionSelector()
        self.q_calculator = QValueCalculator(support)

    def select_action(self, rgb_obs, low_dim_obs):
        """Select action through hierarchical refinement."""
        metrics = {}
        low = _initialize_bounds(rgb_obs, self.initial_low)
        high = _initialize_bounds(rgb_obs, self.initial_high)

        for level in range(self.levels):
            low, high = self._refine_at_level(
                level, rgb_obs, low_dim_obs, low, high, metrics
            )

        continuous_action = (high + low) / 2.0
        return continuous_action, metrics

    def _refine_at_level(self, level, rgb_obs, low_dim_obs, low, high, metrics):
        """Refine action interval at a single level."""
        midpoint = (low + high) / 2
        q_logits = self.network(level, rgb_obs, low_dim_obs, midpoint)
        q_probs = F.softmax(q_logits, dim=3)

        q_values = self.q_calculator.calculate(q_probs)
        bin_index = self._select_bin(q_values)

        _record_metrics(level, q_values, bin_index, metrics)

        return self.interval_zoomer.zoom_in(low, high, bin_index)

    def _select_bin(self, q_values):
        """Select bin index from Q-values."""
        argmax_q = self.random_selector.select_or_argmax(q_values)
        if argmax_q is None:
            argmax_q = q_values.max(-1)[1]
        return argmax_q


def _stack_distributions(q_probs, q_probs_a, log_q_probs, log_q_probs_a):
    """Stack distributions from all levels."""
    return {
        'q_probs': torch.stack(q_probs, dim=-4),
        'q_probs_a': torch.stack(q_probs_a, dim=-3),
        'log_q_probs': torch.stack(log_q_probs, dim=-4),
        'log_q_probs_a': torch.stack(log_q_probs_a, dim=-3)
    }


class C2FCritic(nn.Module):
    """Coarse-to-Fine Critic with distributional Q-learning."""

    def __init__(
            self,
            action_shape: tuple,
            repr_dim: int,
            low_dim: int,
            feature_dim: int,
            hidden_dim: int,
            levels: int,
            bins: int,
            atoms: int,
            v_min: float,
            v_max: float,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.network = C2FCriticNetwork(
            repr_dim,
            low_dim,
            action_shape,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
        )
        self._initialize_parameters(action_shape, atoms, v_min, v_max)
        self._initialize_components(levels, bins)



    def _initialize_parameters(self, action_shape, atoms, v_min, v_max):
        """Initialize learnable parameters."""
        actor_dim = action_shape[0]
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim),
            requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim),
            requires_grad=False
        )
        self.support = nn.Parameter(
            torch.linspace(v_min, v_max, atoms),
            requires_grad=False
        )
        self.delta_z = (v_max - v_min) / (atoms - 1)

    def _initialize_components(self, levels, bins):
        """Initialize critic components."""
        self.action_encoder = ActionEncoder(
            self.initial_low,
            self.initial_high,
            levels,
            bins
        )
        self.action_selector = HierarchicalActionSelector(
            self.network,
            self.initial_low,
            self.initial_high,
            levels,
            bins,
            self.support
        )
        self.projector = DistributionalProjection(
            self.v_min,
            self.v_max,
            self.atoms
        )

    def get_action(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor):
        """Get action through hierarchical coarse-to-fine selection."""
        return self.action_selector.select_action(rgb_obs, low_dim_obs)

    def forward(
            self,
            rgb_obs: torch.Tensor,
            low_dim_obs: torch.Tensor,
            continuous_action: torch.Tensor,
    ):
        """Compute value distributions for given observation and action."""
        discrete_action = self.action_encoder.encode(continuous_action)

        distributions = self._compute_distributions_all_levels(
            rgb_obs, low_dim_obs, discrete_action
        )

        return (
            distributions['q_probs'],
            distributions['q_probs_a'],
            distributions['log_q_probs'],
            distributions['log_q_probs_a']
        )

    def _compute_distributions_all_levels(self, rgb_obs, low_dim_obs, discrete_action):
        """Compute Q-value distributions for all levels."""
        q_probs_per_level = []
        q_probs_a_per_level = []
        log_q_probs_per_level = []
        log_q_probs_a_per_level = []

        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        for level in range(self.levels):
            distributions = self._compute_distributions_single_level(
                level, rgb_obs, low_dim_obs, discrete_action, low, high
            )

            q_probs_per_level.append(distributions['q_probs'])
            q_probs_a_per_level.append(distributions['q_probs_a'])
            log_q_probs_per_level.append(distributions['log_q_probs'])
            log_q_probs_a_per_level.append(distributions['log_q_probs_a'])

            bin_index = discrete_action[..., level, :].long()
            low, high = self.action_selector.interval_zoomer.zoom_in(low, high, bin_index)

        return _stack_distributions(
            q_probs_per_level,
            q_probs_a_per_level,
            log_q_probs_per_level,
            log_q_probs_a_per_level
        )

    def _compute_distributions_single_level(
            self, level, rgb_obs, low_dim_obs, discrete_action, low, high
    ):
        """Compute distributions for a single level."""
        midpoint = (low + high) / 2
        q_logits = self.network(level, rgb_obs, low_dim_obs, midpoint)
        bin_index = discrete_action[..., level, :].long()

        q_probs = F.softmax(q_logits, dim=3)
        q_probs_a = self._gather_action_probs(q_probs, bin_index)

        log_q_probs = F.log_softmax(q_logits, dim=3)
        log_q_probs_a = self._gather_action_probs(log_q_probs, bin_index)

        return {
            'q_probs': q_probs,
            'q_probs_a': q_probs_a,
            'log_q_probs': log_q_probs,
            'log_q_probs_a': log_q_probs_a
        }

    def _gather_action_probs(self, probs, bin_index):
        """Gather probabilities for selected action bins."""
        gathered = torch.gather(
            probs,
            dim=-2,
            index=bin_index.unsqueeze(-1)
            .unsqueeze(-1)
            .repeat_interleave(self.atoms, -1),
        )
        return gathered[..., 0, :]

    def compute_target_q_dist(
            self,
            next_rgb_obs: torch.Tensor,
            next_low_dim_obs: torch.Tensor,
            next_continuous_action: torch.Tensor,
            reward: torch.Tensor,
            discount: torch.Tensor,
    ):
        """Compute target Q-value distribution for distributional RL."""
        next_q_probs_a = self.forward(
            next_rgb_obs, next_low_dim_obs, next_continuous_action
        )[1]

        return self.projector.project(
            next_q_probs_a, reward, discount, self.support
        )

    def encode_decode_action(self, continuous_action: torch.Tensor):
        """Encode continuous action to discrete and decode back."""
        discrete_action = self.action_encoder.encode(continuous_action)
        return self.action_encoder.decode(discrete_action)