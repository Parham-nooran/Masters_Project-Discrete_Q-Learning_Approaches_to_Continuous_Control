"""
Hierarchical coarse-to-fine action discretizer for CQN.
"""

from typing import Dict, Optional

import torch


class CoarseToFineDiscretizer:
    """
    Discretizer for hierarchical coarse-to-fine action selection.

    Creates action bins at each level that progressively refine around
    the parent level's selected action. At level 0, the full action space
    is discretized. At higher levels, the space is refined within the
    range determined by the parent selection.
    """

    def __init__(self, action_spec: Dict, num_levels: int = 3, num_bins: int = 5):
        """
        Initialize discretizer.

        Args:
            action_spec: Dictionary with 'low' and 'high' action bounds.
            num_levels: Number of hierarchy levels.
            num_bins: Number of discrete bins per dimension per level.
        """
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

    def _create_hierarchical_bins(self) -> None:
        """Create bins for level 0 only. Other levels are computed dynamically."""
        self.action_bins = {0: {}}
        for dim in range(self.action_dim):
            bins = torch.linspace(
                self.action_min[dim],
                self.action_max[dim],
                self.num_bins,
                device=self.device,
                dtype=torch.float32,
            )
            self.action_bins[0][dim] = bins
        for level in range(1, self.num_levels):
            self.action_bins[level] = {}

    def get_action_range_for_level(
        self, level: int, parent_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get action range to discretize at current level.

        For level 0, returns full action space. For higher levels,
        returns refined range around parent action selections.

        Args:
            level: Current hierarchy level (0 to num_levels-1).
            parent_actions: Continuous actions from previous level [action_dim].
                           None for level 0.

        Returns:
            Range tensor [2, action_dim] with min and max per dimension.
        """
        if level == 0 or parent_actions is None:
            return torch.stack([self.action_min, self.action_max], dim=0)

        return self._compute_refined_range(level, parent_actions)

    def _compute_refined_range(
        self, level: int, parent_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute refined action range based on parent selections.

        Args:
            level: Current hierarchy level.
            parent_actions: Continuous actions from previous level [action_dim].

        Returns:
            Refined range [2, action_dim].
        """
        parent_range_width = self.action_max - self.action_min
        prev_level_bin_width = parent_range_width / (self.num_bins ** level)

        range_min = parent_actions - prev_level_bin_width / 2
        range_max = parent_actions + prev_level_bin_width / 2

        range_min = torch.clamp(range_min, self.action_min, self.action_max)
        range_max = torch.clamp(range_max, self.action_min, self.action_max)

        return torch.stack([range_min, range_max], dim=0)


    def discrete_to_continuous(
            self, discrete_actions: torch.Tensor, level: int, parent_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert discrete action indices to continuous values.
        For level > 0, parent_action must be provided.
        """
        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if len(discrete_actions.shape) == 1:
            discrete_actions = discrete_actions.unsqueeze(0)

        bin_indices = torch.clamp(discrete_actions.long(), 0, self.num_bins - 1)

        if level == 0:
            bin_width = (self.action_max - self.action_min) / self.num_bins
            continuous_actions = (
                    self.action_min.unsqueeze(0)
                    + bin_indices * bin_width.unsqueeze(0)
                    + bin_width.unsqueeze(0) / 2
            )
        else:
            if parent_action is None:
                raise ValueError(f"parent_action required for level {level}")

            action_range = self.get_action_range_for_level(level, parent_action[0] if len(
                parent_action.shape) > 1 else parent_action)
            range_min, range_max = action_range[0], action_range[1]
            bin_width = (range_max - range_min) / self.num_bins

            continuous_actions = (
                    range_min.unsqueeze(0)
                    + bin_indices * bin_width.unsqueeze(0)
                    + bin_width.unsqueeze(0) / 2
            )

        return continuous_actions.to(self.device)
