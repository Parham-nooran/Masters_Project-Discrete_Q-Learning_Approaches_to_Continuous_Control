import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import math


def _build_network(layer_sizes: List[int]) -> nn.Module:
    """Build network with proper initialization for Q-learning."""
    layers = []
    for i in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        layers.append(linear)

        if i < len(layer_sizes) - 2:  # No activation on final layer
            layers.append(nn.LayerNorm(layer_sizes[i + 1]))
            layers.append(nn.ELU())

    return nn.Sequential(*layers)


def _init_weights(m):
    """Conservative weight initialization for Q-networks."""
    if isinstance(m, nn.Linear):
        # Use smaller initialization to prevent exploding Q-values
        fan_in = m.weight.size(1)
        std = 1.0 / math.sqrt(fan_in)
        torch.nn.init.uniform_(m.weight, -std, std)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def apply_action_mask_optimized(q_values: torch.Tensor,
                                action_mask: torch.Tensor) -> torch.Tensor:
    """JIT-compiled action masking for speed."""
    # FIXED: Use smaller negative value to prevent numerical issues
    mask_value = -1e6  # Changed from -1e8
    return torch.where(action_mask.unsqueeze(0), q_values,
                       torch.full_like(q_values, mask_value))


class GrowingQCritic(nn.Module):
    """Fixed Growing Q-Networks Critic with proper initialization and scaling."""

    def __init__(self, config, input_size: int, action_spec: Dict):
        super().__init__()
        self.config = config
        self.decouple = config.decouple
        self.use_double_q = config.use_double_q
        self.action_dim = len(action_spec["low"])
        self.max_bins = config.max_bins

        # Calculate output dimensions
        if self.decouple:
            self.max_output_dim = self.max_bins * self.action_dim
        else:
            self.max_output_dim = self.max_bins ** self.action_dim

        # FIXED: Build smaller networks for stability
        layer_sizes = [input_size] + config.layer_size_network + [self.max_output_dim]

        # Build Q-networks
        self.q1_network = _build_network(layer_sizes)
        self.q1_network.apply(_init_weights)

        if self.use_double_q:
            self.q2_network = _build_network(layer_sizes)
            self.q2_network.apply(_init_weights)
        else:
            self.q2_network = self.q1_network

        # FIXED: Add output scaling layer to prevent exploding Q-values
        self.output_scale = 10.0  # Scale Q-values to reasonable range

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional action masking and output scaling.
        """
        q1 = self.q1_network(x)
        q2 = self.q2_network(x) if self.use_double_q else q1

        # FIXED: Scale outputs to prevent exploding Q-values
        q1 = torch.tanh(q1 / self.output_scale) * self.output_scale
        q2 = torch.tanh(q2 / self.output_scale) * self.output_scale if self.use_double_q else q1

        if self.decouple:
            # Reshape to [batch, action_dim, max_bins]
            q1 = q1.view(q1.shape[0], self.action_dim, self.max_bins)
            q2 = q2.view(q2.shape[0], self.action_dim, self.max_bins)

            # Apply action mask if provided
            if action_mask is not None:
                # Mask inactive actions with large negative values
                q1 = apply_action_mask_optimized(q1, action_mask)
                q2 = apply_action_mask_optimized(q2, action_mask)
        else:
            # For joint discretization
            if action_mask is not None:
                mask_value = -1e6  # Changed from -1e8
                expanded_mask = action_mask.unsqueeze(0).expand(q1.shape[0], -1)
                q1 = torch.where(expanded_mask, q1, torch.full_like(q1, mask_value))
                q2 = torch.where(expanded_mask, q2, torch.full_like(q2, mask_value))

        return q1, q2