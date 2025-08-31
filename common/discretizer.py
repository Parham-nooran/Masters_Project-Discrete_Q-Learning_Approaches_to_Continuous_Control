import torch

class Discretizer:
    def __init__(self, decouple, action_dim, action_bins, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.decouple = decouple
        self.action_dim = action_dim
        self.action_bins = action_bins


    def discrete_to_continuous(self, discrete_actions: torch.Tensor) -> torch.Tensor:
        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if self.decouple:
            if len(discrete_actions.shape) == 1:
                discrete_actions = discrete_actions.unsqueeze(0)

            batch_size = discrete_actions.shape[0]
            continuous_actions = torch.zeros(batch_size, self.action_dim, device=self.device)

            for dim in range(self.action_dim):
                bin_indices = discrete_actions[:, dim].long()
                continuous_actions[:, dim] = self.action_bins[dim][bin_indices]

            return continuous_actions
        else:
            if len(discrete_actions.shape) == 0:
                discrete_actions = discrete_actions.unsqueeze(0)
            return self.action_bins[discrete_actions.long()]