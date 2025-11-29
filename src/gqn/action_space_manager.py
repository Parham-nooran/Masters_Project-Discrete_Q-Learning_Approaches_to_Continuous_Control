import torch


class ActionSpaceManager:
    """Manages growing action space discretization."""

    def __init__(self, action_spec, initial_bins, final_bins, device):
        self.device = device
        self.action_min = torch.tensor(action_spec["low"], dtype=torch.float32, device=device)
        self.action_max = torch.tensor(action_spec["high"], dtype=torch.float32, device=device)
        self.action_dim = len(self.action_min)

        self.initial_bins = initial_bins
        self.final_bins = final_bins
        self.current_bins = initial_bins

        self.growth_sequence = self._compute_growth_sequence()
        self.current_growth_stage = 0

        self.action_bins = self._create_action_bins()
        self.active_mask = self._create_active_mask()

    def _compute_growth_sequence(self):
        """Compute bin growth sequence: 2->3->5->9 or 9->17->33->65."""
        sequence = [self.initial_bins]
        current = self.initial_bins

        while current < self.final_bins:
            if current == 2:
                current = 3
            elif current < self.final_bins:
                current = current * 2 - 1

            if current > self.final_bins:
                current = self.final_bins

            if current not in sequence:
                sequence.append(current)

        return sequence

    def _create_action_bins(self):
        """Create discretized action bins for all dimensions."""
        bins = []
        for dim in range(self.action_dim):
            dim_bins = torch.linspace(
                self.action_min[dim],
                self.action_max[dim],
                self.final_bins,
                device=self.device
            )
            bins.append(dim_bins)

        return torch.stack(bins)

    def _create_active_mask(self):
        """Create mask for currently active bins."""
        mask = torch.zeros(self.action_dim, self.final_bins, dtype=torch.bool, device=self.device)

        active_indices = self._get_active_bin_indices()
        for dim in range(self.action_dim):
            mask[dim, active_indices] = True

        return mask

    def _get_active_bin_indices(self):
        """Get indices of currently active bins."""
        if self.current_bins == self.final_bins:
            return list(range(self.final_bins))

        step = (self.final_bins - 1) / (self.current_bins - 1)
        return [int(round(i * step)) for i in range(self.current_bins)]

    def grow_action_space(self):
        """Grow action space to next resolution."""
        if self.current_growth_stage < len(self.growth_sequence) - 1:
            self.current_growth_stage += 1
            self.current_bins = self.growth_sequence[self.current_growth_stage]
            self.active_mask = self._create_active_mask()
            return True
        return False

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete action indices to continuous actions."""
        if discrete_actions.device != self.device:
            discrete_actions = discrete_actions.to(self.device)

        if len(discrete_actions.shape) == 1:
            discrete_actions = discrete_actions.unsqueeze(0)

        batch_size = discrete_actions.shape[0]
        continuous_actions = torch.zeros(batch_size, self.action_dim, device=self.device)

        active_indices = self._get_active_bin_indices()

        for dim in range(self.action_dim):
            bin_indices = discrete_actions[:, dim].long()
            bin_indices = torch.clamp(bin_indices, 0, self.current_bins - 1)
            actual_indices = [active_indices[idx] for idx in bin_indices]
            continuous_actions[:, dim] = self.action_bins[dim, actual_indices]

        return continuous_actions

    def get_active_q_values(self, q_values):
        """Extract Q-values for active bins only."""
        batch_size = q_values.shape[0]
        active_indices = self._get_active_bin_indices()

        active_q = torch.zeros(batch_size, self.action_dim, self.current_bins,
                               device=self.device)

        for dim in range(self.action_dim):
            active_q[:, dim, :] = q_values[:, dim, active_indices]

        return active_q

    def can_grow(self):
        """Check if action space can still grow."""
        return self.current_growth_stage < len(self.growth_sequence) - 1

    def get_growth_info(self):
        """Get current growth stage information."""
        return {
            "current_bins": self.current_bins,
            "stage": self.current_growth_stage,
            "total_stages": len(self.growth_sequence),
            "sequence": self.growth_sequence
        }