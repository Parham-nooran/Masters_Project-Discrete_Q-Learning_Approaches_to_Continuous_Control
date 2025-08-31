from typing import Dict

import torch

from common.discretizer import Discretizer


class GrowingActionDiscretizer(Discretizer):
    """
    Action discretizer that supports growing resolution as described in the GQN paper.
    Implements both linear and adaptive growing schedules.
    """

    def __init__(self, action_spec: Dict, num_bins, max_bins: int = 9, decouple: bool = True):
        super().__init__(action_spec, num_bins, decouple)

        self.action_min = torch.tensor(action_spec["low"], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(action_spec["high"], dtype=torch.float32, device=self.device)
        self.action_dim = len(self.action_min)

        # Growing parameters
        self.max_bins = max_bins
        self.current_bins = 2  # Start with 2 bins as per paper
        self.growth_sequence = [2, 3, 5, 9]  # As specified in paper experiments
        self.current_growth_idx = 0

        # Pre-compute all bin levels for efficiency
        self.all_action_bins = {}
        for num_bins in self.growth_sequence:
            if decouple:
                # Per-dimension discretization
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
                # Joint discretization (cartesian product)
                bins_per_dim = []
                for dim in range(self.action_dim):
                    bins = torch.linspace(
                        self.action_min[dim],
                        self.action_max[dim],
                        num_bins,
                        device=self.device
                    )
                    bins_per_dim.append(bins)

                mesh = torch.meshgrid(*bins_per_dim, indexing="ij")
                joint_bins = torch.stack([m.flatten() for m in mesh], dim=1)
                self.all_action_bins[num_bins] = joint_bins

        self.action_bins = self.all_action_bins[self.current_bins]

    def grow_action_space(self) -> bool:
        """
        Grow to next resolution level. Returns True if growth occurred.
        """
        if self.current_growth_idx < len(self.growth_sequence) - 1:
            self.current_growth_idx += 1
            self.current_bins = self.growth_sequence[self.current_growth_idx]
            self.action_bins = self.all_action_bins[self.current_bins]
            return True
        return False

    def get_action_mask(self, full_bins: int) -> torch.Tensor:
        """
        Generate action mask for current active bins within full discretization.
        This implements the action masking mechanism from Figure 1 in the paper.
        """
        if not self.decouple:
            # For joint discretization, mask based on total actions
            total_current = self.current_bins ** self.action_dim
            total_full = full_bins ** self.action_dim
            mask = torch.zeros(total_full, dtype=torch.bool, device=self.device)

            # Calculate which joint actions are active based on current resolution
            step = total_full // total_current
            for i in range(total_current):
                mask[i * step] = True
            return mask
        else:
            # For decoupled case, create mask for each dimension
            mask = torch.zeros(self.action_dim, full_bins, dtype=torch.bool, device=self.device)

            # Calculate step size for current resolution
            step = max(1, full_bins // self.current_bins)
            for dim in range(self.action_dim):
                for i in range(self.current_bins):
                    idx = min(i * step, full_bins - 1)
                    mask[dim, idx] = True

            return mask
