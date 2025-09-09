import torch
from src.common.discretizer import Discretizer


class GrowingActionDiscretizer(Discretizer):
    """Growing action discretizer - minimal implementation matching 2024 paper."""

    def __init__(self, action_spec, max_bins, decouple=True):
        super().__init__(decouple, action_spec, max_bins)

        self.action_spec = action_spec
        self.action_min = torch.tensor(action_spec["low"], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(action_spec["high"], dtype=torch.float32, device=self.device)
        self.action_dim = len(self.action_min)
        self.max_bins = max_bins

        # Growing sequence as per paper: start with 2, grow to 3, 5, 9
        self.growth_sequence = [2, 3, 5, 9]
        self.current_bins = self.growth_sequence[0]  # Start with 2 bins
        self.current_growth_idx = 0

        # Pre-compute all action bins for each resolution level
        self._precompute_action_bins()
        self.action_bins = self.all_action_bins[self.current_bins]

    def _precompute_action_bins(self):
        """Pre-compute action bins for all resolution levels."""
        self.all_action_bins = {}

        for num_bins in self.growth_sequence:
            if self.decouple:
                # Per-dimension discretization (DecQN style)
                bins_per_dim = []
                for dim in range(self.action_dim):
                    bins = torch.linspace(
                        self.action_min[dim],
                        self.action_max[dim],
                        num_bins,
                        device=self.device
                    )
                    bins_per_dim.append(bins)
                self.all_action_bins[num_bins] = torch.stack(bins_per_dim)
            else:
                # Joint discretization - create cartesian product
                bins_per_dim = [
                    torch.linspace(self.action_min[i], self.action_max[i], num_bins, device=self.device)
                    for i in range(self.action_dim)
                ]
                mesh = torch.meshgrid(*bins_per_dim, indexing="ij")
                self.all_action_bins[num_bins] = torch.stack([m.flatten() for m in mesh], dim=1)

    def grow_action_space(self):
        """Grow to next resolution level. Returns True if growth occurred."""
        if self.current_growth_idx < len(self.growth_sequence) - 1:
            self.current_growth_idx += 1
            self.current_bins = self.growth_sequence[self.current_growth_idx]
            self.action_bins = self.all_action_bins[self.current_bins]
            return True
        return False

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete actions to continuous - same as DecQN."""
        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if self.decouple:
            if len(discrete_actions.shape) == 1:
                discrete_actions = discrete_actions.unsqueeze(0)

            batch_size = discrete_actions.shape[0]
            continuous_actions = torch.zeros(batch_size, self.action_dim, device=self.device)

            for dim in range(self.action_dim):
                bin_indices = discrete_actions[:, dim].long()
                bin_indices = torch.clamp(bin_indices, 0, self.current_bins - 1)
                continuous_actions[:, dim] = self.action_bins[dim][bin_indices]

            return continuous_actions
        else:
            if len(discrete_actions.shape) == 0:
                discrete_actions = discrete_actions.unsqueeze(0)
            flat_indices = discrete_actions.long()
            max_joint_actions = len(self.action_bins)
            flat_indices = torch.clamp(flat_indices, 0, max_joint_actions - 1)
            return self.action_bins[flat_indices]