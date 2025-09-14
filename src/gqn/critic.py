from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


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
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def apply_action_mask_optimized(
    q_values: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """JIT-compiled action masking for speed."""
    mask_value = -1e6
    return torch.where(
        action_mask.unsqueeze(0), q_values, torch.full_like(q_values, mask_value)
    )


def _init_final_layer(layer):
    nn.init.uniform_(layer.weight, -0.003, 0.003)
    nn.init.zeros_(layer.bias)


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
            self.max_output_dim = self.max_bins**self.action_dim

        layer_sizes = [input_size] + config.layer_size_network + [self.max_output_dim]

        self.q1_network = _build_network(layer_sizes)
        self.q1_network.apply(_init_weights)

        if self.use_double_q:
            self.q2_network = _build_network(layer_sizes)
            self.q2_network.apply(_init_weights)
        else:
            self.q2_network = self.q1_network

        _init_final_layer(self.q1_network[-1])
        if self.use_double_q:
            _init_final_layer(self.q2_network[-1])

    def forward(
        self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional action masking and output scaling.
        """
        q1 = self.q1_network(x)
        q2 = self.q2_network(x) if self.use_double_q else q1

        if self.decouple:
            q1 = q1.view(q1.shape[0], self.action_dim, self.max_bins)
            q2 = (
                q2.view(q2.shape[0], self.action_dim, self.max_bins)
                if self.use_double_q
                else q1
            )

            if action_mask is not None:
                q1 = apply_action_mask_optimized(q1, action_mask)
                q2 = apply_action_mask_optimized(q2, action_mask)
        else:
            if action_mask is not None:
                mask_value = -1e6
                expanded_mask = action_mask.unsqueeze(0).expand(q1.shape[0], -1)
                q1 = torch.where(expanded_mask, q1, torch.full_like(q1, mask_value))
                q2 = torch.where(expanded_mask, q2, torch.full_like(q2, mask_value))

        return q1, q2
