import torch


class Discretizer:
    """
    Discretizes continuous action spaces for DecQN.

    Handles both decoupled (per-dimension) and coupled (joint) discretization.
    """

    def __init__(self, action_spec, num_bins, decouple):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decouple = decouple
        self.num_bins = num_bins

        self.action_min = torch.tensor(
            action_spec["low"], dtype=torch.float32, device=self.device
        )
        self.action_max = torch.tensor(
            action_spec["high"], dtype=torch.float32, device=self.device
        )
        self.action_dim = len(self.action_min)

        self._initialize_action_bins()

    def _initialize_action_bins(self):
        """Initialize discretized action bins based on decoupling mode."""
        if self.decouple:
            self.action_bins = torch.stack([
                torch.linspace(
                    self.action_min[dim],
                    self.action_max[dim],
                    self.num_bins,
                    device=self.device,
                )
                for dim in range(self.action_dim)
            ])
        else:
            bins_per_dim = [
                torch.linspace(
                    self.action_min[i],
                    self.action_max[i],
                    self.num_bins,
                    device=self.device,
                )
                for i in range(self.action_dim)
            ]
            mesh = torch.meshgrid(*bins_per_dim, indexing="ij")
            self.action_bins = torch.stack([m.flatten() for m in mesh], dim=1)

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

    def _decoupled_discrete_to_continuous(self, discrete_actions):
        """Convert decoupled discrete actions to continuous."""
        if discrete_actions.ndim == 1:
            discrete_actions = discrete_actions.unsqueeze(0)

        batch_size = discrete_actions.shape[0]
        continuous_actions = torch.zeros(
            batch_size, self.action_dim, device=self.device
        )

        for dim in range(self.action_dim):
            bin_indices = discrete_actions[:, dim].long().clamp(0, self.num_bins - 1)
            continuous_actions[:, dim] = self.action_bins[dim, bin_indices]

        return continuous_actions

    def _coupled_discrete_to_continuous(self, discrete_actions):
        """Convert coupled discrete actions to continuous."""
        if discrete_actions.ndim == 0:
            discrete_actions = discrete_actions.unsqueeze(0)

        flat_indices = discrete_actions.long().clamp(0, len(self.action_bins) - 1)
        return self.action_bins[flat_indices]