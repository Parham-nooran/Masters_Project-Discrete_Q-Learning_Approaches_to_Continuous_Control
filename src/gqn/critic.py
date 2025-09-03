import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


def _build_network(layer_sizes: List[int]) -> nn.Module:
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:  # No activation on final layer
            layers.append(nn.LayerNorm(layer_sizes[i + 1]))
            layers.append(nn.ELU())

    return nn.Sequential(*layers)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.4)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class GrowingQCritic(nn.Module):
    def __init__(self, config, input_size: int, action_spec: Dict):
        super().__init__()
        self.config = config
        self.decouple = config.decouple
        self.use_double_q = config.use_double_q
        self.action_dim = len(action_spec["low"])
        self.max_bins = config.max_bins
        if self.decouple:
            self.max_output_dim = self.max_bins * self.action_dim
        else:
            self.max_output_dim = self.max_bins ** self.action_dim
        layer_sizes = [input_size] + config.layer_size_network + [self.max_output_dim]

        self.q1_network = _build_network(layer_sizes)
        self.q1_network.apply(_init_weights)
        if self.use_double_q:
            self.q2_network = _build_network(layer_sizes)
            self.q2_network.apply(_init_weights)
        else:
            self.q2_network = self.q1_network

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional action masking.
        """
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)

        if self.decouple:
            # Reshape to [batch, action_dim, max_bins]
            q1 = q1.view(q1.shape[0], self.action_dim, self.max_bins)
            q2 = q2.view(q2.shape[0], self.action_dim, self.max_bins)

            # Apply action mask if provided
            if action_mask is not None:
                # Mask inactive actions with large negative values
                mask_value = -1e8
                expanded_mask = action_mask.unsqueeze(0).expand(q1.shape[0], -1, -1)
                q1 = torch.where(expanded_mask, q1, torch.full_like(q1, mask_value))
                q2 = torch.where(expanded_mask, q2, torch.full_like(q2, mask_value))
        else:
            # For joint discretization
            if action_mask is not None:
                mask_value = -1e8
                expanded_mask = action_mask.unsqueeze(0).expand(q1.shape[0], -1)
                q1 = torch.where(expanded_mask, q1, torch.full_like(q1, mask_value))
                q2 = torch.where(expanded_mask, q2, torch.full_like(q2, mask_value))

        return q1, q2


