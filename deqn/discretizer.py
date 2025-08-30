from replay_buffer import *


class ActionDiscretizer:
    """Handles continuous to discrete action conversion."""

    def __init__(self, action_spec, num_bins, decouple=False):
        self.num_bins = num_bins
        self.decouple = decouple
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete actions to continuous."""
        if not isinstance(discrete_actions, torch.Tensor):
            discrete_actions = torch.tensor(discrete_actions, device=self.device)

        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if self.decouple:
            # discrete_actions shape: [batch_size, action_dim]
            if len(discrete_actions.shape) == 1:
                discrete_actions = discrete_actions.unsqueeze(0)

            batch_size = discrete_actions.shape[0]
            continuous_actions = torch.zeros(
                batch_size, self.action_dim, device=self.device
            )

            for dim in range(self.action_dim):
                bin_indices = discrete_actions[:, dim].long()
                continuous_actions[:, dim] = self.action_bins[dim][bin_indices]

            return continuous_actions
        else:
            if len(discrete_actions.shape) == 0:
                discrete_actions = discrete_actions.unsqueeze(0)
            return self.action_bins[discrete_actions.long()]


