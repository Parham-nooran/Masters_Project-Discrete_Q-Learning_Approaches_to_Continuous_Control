import itertools
from typing import Dict

import torch

from src.common.discretizer import Discretizer


class GrowingActionDiscretizer(Discretizer):
    """
    Action discretizer that supports growing resolution as described in the GQN paper.
    Implements both linear and adaptive growing schedules.
    """

    def __init__(self, action_spec: Dict, max_bins: int = 9, decouple: bool = True,
                 growth_sequence: list = None):
        super().__init__(decouple, action_spec, max_bins)
        self.action_min = torch.tensor(action_spec["low"], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(action_spec["high"], dtype=torch.float32, device=self.device)
        self.action_dim = len(self.action_min)

        # Growing parameters
        self.max_bins = max_bins
        self.growth_sequence = growth_sequence if growth_sequence is not None else [2, 3, 5,
                                                                                    9]  # As specified in paper experiments
        self.current_bins = self.growth_sequence[0]  # Start with 2 bins as per paper
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
            # For joint discretization
            total_full = full_bins ** self.action_dim
            mask = torch.zeros(total_full, dtype=torch.bool, device=self.device)

            # Calculate step size for uniform sampling across resolution levels
            if full_bins >= self.current_bins:
                step = full_bins // self.current_bins
                if step * self.current_bins == full_bins:
                    # Perfect divisibility
                    indices = list(range(0, full_bins, step))
                else:
                    # Use linear interpolation for non-divisible cases
                    indices = list(dict.fromkeys([int(round(i * (full_bins - 1) / (self.current_bins - 1)))
                                                  for i in range(self.current_bins)]))
            else:
                # Current bins exceed full bins - use all available
                indices = list(range(full_bins))

            # Generate all valid combinations for joint discretization
            for combo in itertools.product(indices, repeat=self.action_dim):
                # Convert multi-dimensional index to flat index
                flat_idx = 0
                for i, idx in enumerate(combo):
                    flat_idx += idx * (full_bins ** (self.action_dim - 1 - i))

                if flat_idx < total_full:
                    mask[flat_idx] = True

            return mask
        else:
            # For decoupled case - per-dimension masking
            mask = torch.zeros(self.action_dim, full_bins, dtype=torch.bool, device=self.device)

            if full_bins >= self.current_bins:
                step = full_bins // self.current_bins
                if step * self.current_bins == full_bins:
                    # Perfect divisibility
                    indices = torch.arange(0, full_bins, step, dtype=torch.long, device=self.device)
                else:
                    # Use linear interpolation for non-divisible cases
                    indices = torch.round(
                        torch.linspace(0, full_bins - 1, self.current_bins, device=self.device)
                    ).long()
            else:
                # Current bins exceed full bins - use all available
                indices = torch.arange(0, full_bins, dtype=torch.long, device=self.device)

            # Apply mask to all dimensions
            for dim in range(self.action_dim):
                valid_indices = torch.clamp(indices, 0, full_bins - 1)
                mask[dim, valid_indices] = True

            return mask