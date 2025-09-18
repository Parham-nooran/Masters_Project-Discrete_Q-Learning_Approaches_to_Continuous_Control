from src.common.discretizer import Discretizer
from src.common.replay_buffer import *


class ActionDiscretizer(Discretizer):
    """Handles continuous to discrete action conversion."""

    def __init__(self, num_bins, action_spec, action_bins, decouple=False):
        self.action_bins = action_bins
        super().__init__(decouple, action_spec, num_bins)
        self.num_bins = num_bins
        self.decouple = decouple

        if decouple:
            self.action_bins = []
            for dim in range(self.action_dim):
                bins = torch.linspace(
                    self.action_min[dim],
                    self.action_max[dim],
                    num_bins,
                    device=self.device,
                )
                self.action_bins.append(bins)
            self.action_bins = torch.stack(self.action_bins)
        else:
            bins_per_dim = torch.stack(
                [
                    torch.linspace(
                        self.action_min[i],
                        self.action_max[i],
                        num_bins,
                        device=self.device,
                    )
                    for i in range(self.action_dim)
                ]
            )
            mesh = torch.meshgrid(*bins_per_dim, indexing="ij")
            self.action_bins = torch.stack([m.flatten() for m in mesh], dim=1)
