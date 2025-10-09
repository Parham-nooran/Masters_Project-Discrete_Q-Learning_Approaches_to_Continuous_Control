from typing import Dict, Optional

import torch


class CoarseToFineDiscretizer:
    """
    Discretizer that creates a hierarchical coarse-to-fine action space.
    Each level has progressively finer discretization.
    """

    def __init__(self, action_spec: Dict, num_levels: int = 3, num_bins: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_spec = action_spec
        self.num_levels = num_levels
        self.num_bins = num_bins
        self.action_min = torch.tensor(
            action_spec["low"], dtype=torch.float32, device=self.device
        )
        self.action_max = torch.tensor(
            action_spec["high"], dtype=torch.float32, device=self.device
        )
        self.action_dim = len(self.action_min)
        self.action_bins = {}
        self._create_hierarchical_bins()

    def _create_hierarchical_bins(self):
        """Create hierarchical bins for each level and dimension"""
        for level in range(self.num_levels):
            self.action_bins[level] = {}
            for dim in range(self.action_dim):
                bins = torch.linspace(
                    self.action_min[dim],
                    self.action_max[dim],
                    self.num_bins,
                    device=self.device,
                    dtype=torch.float32
                )
                self.action_bins[level][dim] = bins

    def get_action_range_for_level(
            self, level: int, parent_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the action range for a specific level.
        If parent_actions is provided, refine the range based on parent selections.
        """
        if level == 0 or parent_actions is None:
            return torch.stack([self.action_min, self.action_max], dim=0)

        ranges = []
        for dim in range(self.action_dim):
            parent_bin_idx = parent_actions[dim].long()

            # parent_bins = self.action_bins[level - 1][dim]
            # bin_width = (parent_bins[-1] - parent_bins[0]) / (self.num_bins - 1)
            # parent_center = parent_bins[parent_bin_idx]
            full_range = self.action_max[dim] - self.action_min[dim]
            bin_width = full_range / self.num_bins
            for _ in range(level):
                bin_width = bin_width / self.num_bins
            parent_center = parent_actions[dim]

            range_min = parent_center - bin_width / 2
            range_max = parent_center + bin_width / 2
            range_min = torch.clamp(
                range_min, self.action_min[dim], self.action_max[dim]
            )
            range_max = torch.clamp(
                range_max, self.action_min[dim], self.action_max[dim]
            )

            ranges.append(torch.stack([range_min, range_max]))
        return torch.stack(ranges, dim=1)

    def discrete_to_continuous(
            self, discrete_actions: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Convert discrete actions to continuous actions for a specific level"""
        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)
        if len(discrete_actions.shape) == 1:
            discrete_actions = discrete_actions.unsqueeze(0)
        batch_size = discrete_actions.shape[0]
        continuous_actions = torch.zeros(
            batch_size, self.action_dim, device=self.device
        )
        for dim in range(self.action_dim):
            bin_indices = discrete_actions[:, dim].long()
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
            bin_width = (self.action_max[dim] - self.action_min[dim]) / self.num_bins
            continuous_actions[:, dim] = self.action_min[dim] + bin_indices * bin_width + bin_width / 2
        return continuous_actions
