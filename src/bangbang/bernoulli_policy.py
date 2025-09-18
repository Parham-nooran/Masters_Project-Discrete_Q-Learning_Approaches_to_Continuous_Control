import logging
import os
import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from dm_control import suite

from agent import BangBangAgent
from src.common.metrics_tracker import MetricsTracker
from src.common.networks import LayerNormMLP


class BernoulliPolicy(nn.Module):
    """Bernoulli policy for bang-bang control as described in the paper."""

    def __init__(
        self, input_size: int, action_dim: int, hidden_sizes: list = [512, 512]
    ):
        super().__init__()
        self.action_dim = action_dim
        sizes = [input_size] + hidden_sizes + [action_dim]
        self.network = LayerNormMLP(sizes, activate_final=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits for Bernoulli distribution."""
        return self.network(obs)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from Bernoulli policy."""
        logits = self.forward(obs)
        probs = torch.sigmoid(logits)

        if deterministic:
            actions = (probs > 0.5).float()
        else:
            dist = torch.distributions.Bernoulli(probs)
            actions = dist.sample()

        # Convert to bang-bang actions: 0 -> -1, 1 -> +1
        bang_bang_actions = 2.0 * actions - 1.0
        log_probs = torch.distributions.Bernoulli(probs).log_prob(actions).sum(dim=-1)

        return bang_bang_actions, log_probs
