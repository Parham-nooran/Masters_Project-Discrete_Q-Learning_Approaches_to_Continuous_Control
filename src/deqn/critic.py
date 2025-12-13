import torch.nn as nn
from src.common.networks import LayerNormAndResidualMLP, LayerNormMLP


def _extract_action_dims_from_dict(action_spec):
    """Extract action dimensionality from dictionary specification."""
    return len(action_spec["low"])


def _extract_action_dims_from_shape(action_spec):
    """Extract action dimensionality from shape attribute."""
    return action_spec.shape[0]


def _extract_action_dims_from_length(action_spec):
    """Extract action dimensionality from length."""
    return len(action_spec)


def _get_action_dims(action_spec):
    """Extract action dimensionality from action spec."""
    if isinstance(action_spec, dict) and "low" in action_spec:
        return _extract_action_dims_from_dict(action_spec)

    if hasattr(action_spec, "shape") and len(action_spec.shape) > 0:
        return _extract_action_dims_from_shape(action_spec)

    if hasattr(action_spec, "__len__"):
        return _extract_action_dims_from_length(action_spec)

    return 1


def _create_residual_block(config):
    """Create residual block layer."""
    return LayerNormAndResidualMLP(config.layer_size_network[0], num_blocks=1)


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
        self.action_dims = _get_action_dims(action_spec)
        self.output_dim = self._compute_output_dim()

        self.q1_network = self._build_network(config, input_size)
        self.q2_network = self._build_network(config, input_size)

    def _compute_decoupled_output_dim(self):
        """Compute output dimension for decoupled mode (linear in action dims)."""
        return self.num_bins * self.action_dims

    def _compute_coupled_output_dim(self):
        """Compute output dimension for coupled mode (exponential in action dims)."""
        return self.num_bins**self.action_dims

    def _compute_output_dim(self):
        """
        Compute output dimension based on discretization mode.

        Decoupled: M * num_bins (linear in action dims)
        Coupled: num_bins^M (exponential in action dims)
        """
        if self.decouple:
            return self._compute_decoupled_output_dim()
        return self._compute_coupled_output_dim()

    def _should_use_residual_network(self, config):
        """Determine if residual network should be used."""
        return config.use_residual and not config.use_pixels

    def _build_network(self, config, input_size):
        """Build Q-network architecture."""
        if self._should_use_residual_network(config):
            return self._build_residual_network(config)

        if config.use_pixels:
            return self._build_pixel_network(config, input_size)

        return self._build_state_network(config, input_size)

    def _create_output_layer(self, input_dim):
        """Create output layer."""
        return nn.Linear(input_dim, self.output_dim)

    def _build_residual_network(self, config):
        """Build network with residual connections."""
        return nn.Sequential(
            nn.Flatten(),
            _create_residual_block(config),
            nn.ELU(),
            self._create_output_layer(config.layer_size_network[0]),
        )

    def _create_layer_sizes(self, input_size, config):
        """Create list of layer sizes for MLP."""
        return [input_size] + config.layer_size_network + [self.output_dim]

    def _build_pixel_network(self, config, input_size):
        """Build network for pixel observations."""
        sizes = self._create_layer_sizes(input_size, config)
        return LayerNormMLP(sizes)

    def _build_state_network(self, config, input_size):
        """Build network for state observations."""
        sizes = self._create_layer_sizes(input_size, config)
        return nn.Sequential(nn.Flatten(), LayerNormMLP(sizes))

    def _reshape_for_decoupled_mode(self, q_values):
        """Reshape Q-values for decoupled action representation."""
        return q_values.view(q_values.shape[0], self.action_dims, self.num_bins)

    def _process_q1_output(self, q1):
        """Process Q1 network output based on mode."""
        if self.decouple:
            return self._reshape_for_decoupled_mode(q1)
        return q1

    def _process_q2_output(self, q2):
        """Process Q2 network output based on mode."""
        if not self.use_double_q:
            return None

        if self.decouple:
            return self._reshape_for_decoupled_mode(q2)
        return q2

    def _compute_q1(self, x):
        """Compute Q1 values."""
        return self.q1_network(x)

    def _compute_q2(self, x, q1):
        """Compute Q2 values or return Q1 if not using double Q."""
        if self.use_double_q:
            return self.q2_network(x)
        return q1

    def forward(self, x):
        """
        Forward pass through both Q-networks.

        Returns:
            q1, q2: Q-values for all actions (decoupled or coupled)
                    Shape: [batch, action_dims, num_bins] if decouple
                           [batch, num_bins^action_dims] if coupled
        """
        q1 = self._compute_q1(x)
        q2 = self._compute_q2(x, q1)

        q1_processed = self._process_q1_output(q1)
        q2_processed = self._process_q2_output(q2)

        return q1_processed, q2_processed if q2_processed is not None else q1_processed
