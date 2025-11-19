from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


def _build_network(layer_sizes: List[int]) -> nn.Module:
    """Build network with proper initialization for Q-learning."""
    layers = []
    for i in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        layers.append(linear)

        if i < len(layer_sizes) - 2:
            layers.append(nn.LayerNorm(layer_sizes[i + 1]))
            layers.append(nn.ELU())

    return nn.Sequential(*layers)


def _init_weights(m, seed=None):
    """Conservative weight initialization for Q-networks with optional seed."""
    if isinstance(m, nn.Linear):
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _init_final_layer(layer, seed=None):
    """Initialize final layer with small weights."""
    if seed is not None:
        torch.manual_seed(seed)
    nn.init.uniform_(layer.weight, -0.003, 0.003)
    nn.init.zeros_(layer.bias)


@torch.jit.script
def apply_action_mask_optimized(
    q_values: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """JIT-compiled action masking for speed."""
    mask_value = torch.tensor(-1e6, dtype=q_values.dtype, device=q_values.device)
    return torch.where(action_mask.unsqueeze(0), q_values, mask_value)


class GrowingQCritic(nn.Module):
    """Growing Q-Critic with decorrelated double Q-network initialization."""

    def __init__(self, config, input_size: int, action_spec: Dict, init_seed: Optional[int] = None):
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
        self.q1_network.apply(lambda m: _init_weights(m, seed=init_seed))
        _init_final_layer(self.q1_network[-1], seed=init_seed)

        if self.use_double_q:
            q2_seed = init_seed + 1000 if init_seed is not None else None
            self.q2_network = _build_network(layer_sizes)
            self.q2_network.apply(lambda m: _init_weights(m, seed=q2_seed))
            _init_final_layer(self.q2_network[-1], seed=q2_seed)
        else:
            self.q2_network = self.q1_network

    def forward(
        self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional action masking."""
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
                mask_value = torch.tensor(-1e6, dtype=q1.dtype, device=q1.device)
                expanded_mask = action_mask.unsqueeze(0).expand_as(q1)
                q1 = torch.where(expanded_mask, q1, mask_value)
                q2 = torch.where(expanded_mask, q2, mask_value)

        return q1, q2