import torch


class Discretizer:
    """
    Discretizes continuous action spaces for DecQN.

    Handles both decoupled (per-dimension) and coupled (joint) discretization.
    """

    def __init__(self, action_spec, num_bins, decouple):
        self.device = self._initialize_device()
        self.decouple = decouple
        self.num_bins = num_bins

        self.action_min = self._create_action_bounds_tensor(action_spec["low"])
        self.action_max = self._create_action_bounds_tensor(action_spec["high"])
        self.action_dim = len(self.action_min)

        self._initialize_action_bins()

    def _initialize_device(self):
        """Initialize computation device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_action_bounds_tensor(self, bounds):
        """Create tensor for action bounds."""
        return torch.tensor(bounds, dtype=torch.float32, device=self.device)

    def _initialize_action_bins(self):
        """Initialize discretized action bins based on decoupling mode."""
        if self.decouple:
            self.action_bins = self._create_decoupled_bins()
        else:
            self.action_bins = self._create_coupled_bins()

    def _create_bins_for_dimension(self, dim):
        """Create discretized bins for a single action dimension."""
        return torch.linspace(
            self.action_min[dim],
            self.action_max[dim],
            self.num_bins,
            device=self.device,
        )

    def _create_decoupled_bins(self):
        """Create separate bins for each action dimension."""
        bins_per_dim = [
            self._create_bins_for_dimension(dim) for dim in range(self.action_dim)
        ]
        return torch.stack(bins_per_dim)

    def _create_meshgrid_from_bins(self, bins_per_dim):
        """Create meshgrid from per-dimension bins."""
        return torch.meshgrid(*bins_per_dim, indexing="ij")

    def _flatten_meshgrid(self, mesh):
        """Flatten meshgrid into action bins."""
        return torch.stack([m.flatten() for m in mesh], dim=1)

    def _create_coupled_bins(self):
        """Create joint bins for all action dimensions."""
        bins_per_dim = [
            self._create_bins_for_dimension(i) for i in range(self.action_dim)
        ]
        mesh = self._create_meshgrid_from_bins(bins_per_dim)
        return self._flatten_meshgrid(mesh)

    def discrete_to_continuous(self, discrete_actions):
        """
        Convert discrete action indices to continuous actions.

        Args:
            discrete_actions: Tensor of discrete action indices

        Returns:
            Continuous actions tensor
        """
        discrete_actions = discrete_actions.to(self.device)

        if self.decouple:
            return self._decoupled_discrete_to_continuous(discrete_actions)
        return self._coupled_discrete_to_continuous(discrete_actions)

    def _ensure_2d_actions(self, discrete_actions):
        """Ensure discrete actions have batch dimension."""
        if discrete_actions.ndim == 1:
            return discrete_actions.unsqueeze(0)
        return discrete_actions

    def _create_empty_continuous_actions(self, batch_size):
        """Create empty tensor for continuous actions."""
        return torch.zeros(batch_size, self.action_dim, device=self.device)

    def _clamp_bin_indices(self, bin_indices):
        """Clamp bin indices to valid range."""
        return bin_indices.long().clamp(0, self.num_bins - 1)

    def _map_indices_to_actions(self, discrete_actions, continuous_actions):
        """Map discrete indices to continuous actions for each dimension."""
        for dim in range(self.action_dim):
            bin_indices = self._clamp_bin_indices(discrete_actions[:, dim])
            continuous_actions[:, dim] = self.action_bins[dim, bin_indices]

    def _decoupled_discrete_to_continuous(self, discrete_actions):
        """Convert decoupled discrete actions to continuous."""
        discrete_actions = self._ensure_2d_actions(discrete_actions)
        batch_size = discrete_actions.shape[0]
        continuous_actions = self._create_empty_continuous_actions(batch_size)
        self._map_indices_to_actions(discrete_actions, continuous_actions)
        return continuous_actions

    def _ensure_1d_actions(self, discrete_actions):
        """Ensure discrete actions are at least 1-dimensional."""
        if discrete_actions.ndim == 0:
            return discrete_actions.unsqueeze(0)
        return discrete_actions

    def _clamp_flat_indices(self, flat_indices):
        """Clamp flat indices to valid range."""
        return flat_indices.long().clamp(0, len(self.action_bins) - 1)

    def _coupled_discrete_to_continuous(self, discrete_actions):
        """Convert coupled discrete actions to continuous."""
        discrete_actions = self._ensure_1d_actions(discrete_actions)
        flat_indices = self._clamp_flat_indices(discrete_actions)
        return self.action_bins[flat_indices]
