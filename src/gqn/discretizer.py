import torch

from src.common.discretizer import Discretizer


class GrowingActionDiscretizer(Discretizer):
    """Growing action discretizer - minimal implementation matching 2024 paper."""

    def __init__(self, action_spec, max_bins, decouple=True):
        self.current_growth_idx = 0
        self.decouple = decouple
        self.max_bins = max_bins
        self.action_spec = action_spec
        self.growth_sequence = [2, 3, 5, 9] if max_bins <= 9 else [2, 3, 5, 9, 17, 33, 65]
        self.num_bins = self.growth_sequence[0]
        super().__init__(decouple, action_spec, self.num_bins)
        self._precompute_action_bins()
        self.action_bins = self.all_action_bins[self.num_bins]

    def _precompute_action_bins(self):
        """Pre-compute action bins for all resolution levels."""
        self.all_action_bins = {}
        for num_bins in self.growth_sequence:
            if self.decouple:
                bins_per_dim = []
                for dim in range(self.action_dim):
                    bins = torch.linspace(
                        self.action_min[dim],
                        self.action_max[dim],
                        num_bins,
                        device=self.device,
                    )
                    bins_per_dim.append(bins)
                self.all_action_bins[num_bins] = torch.stack(bins_per_dim)
            else:
                bins_per_dim = [
                    torch.linspace(
                        self.action_min[i],
                        self.action_max[i],
                        num_bins,
                        device=self.device,
                    )
                    for i in range(self.action_dim)
                ]
                mesh = torch.meshgrid(*bins_per_dim, indexing="ij")
                self.all_action_bins[num_bins] = torch.stack(
                    [m.flatten() for m in mesh], dim=1
                )

    def grow_action_space(self):
        """Grow to next resolution level. Returns True if growth occurred."""
        if self.current_growth_idx < len(self.growth_sequence) - 1:
            self.current_growth_idx += 1
            old_bins = self.num_bins
            self.num_bins = self.growth_sequence[self.current_growth_idx]
            if self.num_bins not in self.all_action_bins:
                self.current_growth_idx -= 1
                self.num_bins = old_bins
                return False
            self.action_bins = self.all_action_bins[self.num_bins]
            return True
        return False