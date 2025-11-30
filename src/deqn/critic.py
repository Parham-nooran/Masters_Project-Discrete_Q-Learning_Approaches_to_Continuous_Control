import torch.nn as nn
from src.common.networks import LayerNormAndResidualMLP, LayerNormMLP


class CriticDQN(nn.Module):
    """
    Double Q-network critic for DecQN.

    Implements decoupled Q-value prediction where each action dimension
    is evaluated independently following the paper's Equation 2.
    """

    def __init__(self, config, input_size, action_spec):
        super().__init__()
        self.use_double_q = config.use_double_q
        self.decouple = config.decouple
        self.num_bins = config.num_bins

        self.action_dims = self._get_action_dims(action_spec)
        self.output_dim = self._compute_output_dim()

        self.q1_network = self._build_network(config, input_size)
        self.q2_network = self._build_network(config, input_size)

    def _get_action_dims(self, action_spec):
        """Extract action dimensionality from action spec."""
        if isinstance(action_spec, dict) and "low" in action_spec:
            return len(action_spec["low"])
        if hasattr(action_spec, "shape") and len(action_spec.shape) > 0:
            return action_spec.shape[0]
        if hasattr(action_spec, "__len__"):
            return len(action_spec)
        return 1

    def _compute_output_dim(self):
        """
        Compute output dimension based on discretization mode.

        Decoupled: M * num_bins (linear in action dims)
        Coupled: num_bins^M (exponential in action dims)
        """
        if self.decouple:
            return self.num_bins * self.action_dims
        return self.num_bins ** self.action_dims

    def _build_network(self, config, input_size):
        """Build Q-network architecture."""
        if config.use_residual and not config.use_pixels:
            return self._build_residual_network(config)

        if config.use_pixels:
            return self._build_pixel_network(config, input_size)

        return self._build_state_network(config, input_size)

    def _build_residual_network(self, config):
        """Build network with residual connections."""
        return nn.Sequential(
            nn.Flatten(),
            LayerNormAndResidualMLP(config.layer_size_network[0], num_blocks=1),
            nn.ELU(),
            nn.Linear(config.layer_size_network[0], self.output_dim),
        )

    def _build_pixel_network(self, config, input_size):
        """Build network for pixel observations."""
        sizes = [input_size] + config.layer_size_network + [self.output_dim]
        return LayerNormMLP(sizes)

    def _build_state_network(self, config, input_size):
        """Build network for state observations."""
        sizes = [input_size] + config.layer_size_network + [self.output_dim]
        return nn.Sequential(nn.Flatten(), LayerNormMLP(sizes))

    def forward(self, x):
        """
        Forward pass through both Q-networks.

        Returns:
            q1, q2: Q-values for all actions (decoupled or coupled)
                    Shape: [batch, action_dims, num_bins] if decouple
                           [batch, num_bins^action_dims] if coupled
        """
        q1 = self.q1_network(x)
        q2 = self.q2_network(x) if self.use_double_q else q1

        if self.decouple:
            q1 = q1.view(q1.shape[0], self.action_dims, self.num_bins)
            q2 = q2.view(q2.shape[0], self.action_dims, self.num_bins) \
                if self.use_double_q else q1

        return q1, q2