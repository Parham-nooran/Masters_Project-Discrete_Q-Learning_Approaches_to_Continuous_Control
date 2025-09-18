import torch


class Discretizer:
    def __init__(self, decouple, action_spec, num_bins):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decouple = decouple
        self.action_spec = action_spec
        self.num_bins = num_bins
        self.action_min = torch.tensor(
            action_spec["low"], dtype=torch.float32, device=self.device
        )
        self.action_max = torch.tensor(
            action_spec["high"], dtype=torch.float32, device=self.device
        )
        self.action_dim = len(self.action_min)

    def discrete_to_continuous(self, discrete_actions: torch.Tensor) -> torch.Tensor:
        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if self.decouple:
            if len(discrete_actions.shape) == 1:
                discrete_actions = discrete_actions.unsqueeze(0)

            batch_size = discrete_actions.shape[0]
            continuous_actions = torch.zeros(
                batch_size, self.action_dim, device=self.device
            )

            for dim in range(self.action_dim):
                bin_indices = discrete_actions[:, dim].long()
                bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
                continuous_actions[:, dim] = self.action_bins[dim][bin_indices]

            return continuous_actions
        else:
            if len(discrete_actions.shape) == 0:
                discrete_actions = discrete_actions.unsqueeze(0)
            flat_indices = discrete_actions.long()
            max_joint_actions = len(self.action_bins)
            flat_indices = torch.clamp(flat_indices, 0, max_joint_actions - 1)
            return self.action_bins[flat_indices]
