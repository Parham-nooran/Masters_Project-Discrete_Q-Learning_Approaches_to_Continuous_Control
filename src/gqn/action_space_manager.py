import torch


class ActionSpaceManager:
    """Manages growing action space discretization with progressive refinement."""

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
            current = self._compute_next_bin_count(current)
            current = min(current, self.final_bins)

            if current not in sequence:
                sequence.append(current)

        return sequence

    def _compute_next_bin_count(self, current):
        """Compute next bin count in growth sequence."""
        if current == 2:
            return 3
        return current * 2 - 1

    def _create_action_bins(self):
        """Create discretized action bins for all dimensions."""
        bins = [self._create_bins_for_dimension(dim) for dim in range(self.action_dim)]
        return torch.stack(bins)

    def _create_bins_for_dimension(self, dim):
        """Create bins for a single action dimension."""
        return torch.linspace(
            self.action_min[dim],
            self.action_max[dim],
            self.final_bins,
            device=self.device
        )

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
        if not self._can_advance_growth_stage():
            return False

        self._advance_to_next_growth_stage()
        self._update_active_bins()
        self._cache_active_indices()
        return True

    def _can_advance_growth_stage(self):
        """Check if growth stage can be advanced."""
        return self.current_growth_stage < len(self.growth_sequence) - 1

    def _advance_to_next_growth_stage(self):
        """Advance to next growth stage."""
        self.current_growth_stage += 1
        self.current_bins = self.growth_sequence[self.current_growth_stage]

    def _update_active_bins(self):
        """Update active bins mask."""
        self.active_mask = self._create_active_mask()

    def _cache_active_indices(self):
        """Cache active indices as tensor for efficient lookups."""
        self._active_indices_tensor = torch.tensor(
            self._get_active_bin_indices(),
            dtype=torch.long,
            device=self.device
        )

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete action indices to continuous actions (vectorized)."""
        discrete_actions = self._ensure_batch_dimension(discrete_actions)
        discrete_actions = self._ensure_on_device(discrete_actions)
        discrete_actions = self._clamp_to_valid_range(discrete_actions)

        actual_indices = self._map_to_actual_bin_indices(discrete_actions)
        continuous_actions = self._gather_continuous_values(actual_indices, discrete_actions.shape[0])

        return continuous_actions

    def _ensure_batch_dimension(self, discrete_actions):
        """Ensure discrete actions have batch dimension."""
        if len(discrete_actions.shape) == 1:
            return discrete_actions.unsqueeze(0)
        return discrete_actions

    def _ensure_on_device(self, discrete_actions):
        """Ensure discrete actions are on correct device."""
        if discrete_actions.device != self.device:
            return discrete_actions.to(self.device)
        return discrete_actions

    def _clamp_to_valid_range(self, discrete_actions):
        """Clamp discrete actions to valid range."""
        return torch.clamp(discrete_actions, 0, self.current_bins - 1)

    def _map_to_actual_bin_indices(self, discrete_actions):
        """Map discrete action indices to actual bin indices."""
        self._ensure_active_indices_cached()
        return self._active_indices_tensor[discrete_actions]

    def _ensure_active_indices_cached(self):
        """Ensure active indices tensor is cached."""
        if not hasattr(self, '_active_indices_tensor'):
            self._cache_active_indices()

    def _gather_continuous_values(self, actual_indices, batch_size):
        """Gather continuous values from bins using actual indices."""
        continuous_actions = torch.gather(
            self.action_bins.unsqueeze(0).expand(batch_size, -1, -1),
            2,
            actual_indices.unsqueeze(2)
        ).squeeze(2)
        return continuous_actions

    def get_active_q_values(self, q_values):
        """Extract Q-values for active bins only (vectorized)."""
        batch_size = q_values.shape[0]
        self._ensure_active_indices_cached()

        active_indices = self._expand_active_indices_for_batch(batch_size)
        active_q = torch.gather(q_values, 2, active_indices)

        return active_q

    def _expand_active_indices_for_batch(self, batch_size):
        """Expand active indices for batch processing."""
        return self._active_indices_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.action_dim, -1
        )

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