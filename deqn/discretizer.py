from common.discretizer import Discretizer
from common.replay_buffer import *


class ActionDiscretizer(Discretizer):
    """Handles continuous to discrete action conversion."""

    def __init__(self, action_spec, num_bins, action_dim, action_bins, decouple=False):
        super().__init__(decouple, action_dim, action_bins)
        self.num_bins = num_bins
        self.decouple = decouple

        # Handle different action_spec formats
        if isinstance(action_spec, dict):
            if "low" in action_spec and "high" in action_spec:
                self.action_min = torch.tensor(
                    action_spec["low"], dtype=torch.float32, device=self.device
                )
                self.action_max = torch.tensor(
                    action_spec["high"], dtype=torch.float32, device=self.device
                )
            else:
                raise ValueError(f"Invalid action_spec format: {action_spec}")
        elif hasattr(action_spec, "low") and hasattr(action_spec, "high"):
            # Handle gym-style Box spaces
            self.action_min = torch.tensor(
                action_spec.low, dtype=torch.float32, device=self.device
            )
            self.action_max = torch.tensor(
                action_spec.high, dtype=torch.float32, device=self.device
            )
        else:
            raise ValueError(f"Unsupported action_spec format: {type(action_spec)}")

        self.action_dim = len(self.action_min)

        if decouple:
            # Per-dimension discretization - create bins for each dimension separately
            self.action_bins = []
            for dim in range(self.action_dim):
                bins = torch.linspace(
                    self.action_min[dim],
                    self.action_max[dim],
                    num_bins,
                    device=self.device,
                )
                self.action_bins.append(bins)
            self.action_bins = torch.stack(
                self.action_bins
            )  # Shape: [action_dim, num_bins]
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
            # Create cartesian product more efficiently
            mesh = torch.meshgrid(*bins_per_dim, indexing="ij")
            self.action_bins = torch.stack([m.flatten() for m in mesh], dim=1)



