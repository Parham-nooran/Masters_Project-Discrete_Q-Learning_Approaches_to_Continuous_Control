from typing import Tuple

import torch
import torch.nn as nn

from src.common.networks import LayerNormMLP


class BernoulliPolicy(nn.Module):

    def __init__(self, input_size: int, action_dim: int, hidden_sizes: list = [512, 512]):
        super().__init__()
        self.action_dim = action_dim
        sizes = [input_size] + hidden_sizes + [action_dim]
        self.network = LayerNormMLP(sizes, activate_final=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        probs = torch.sigmoid(logits)

        if deterministic:
            actions = self._get_deterministic_action(probs)
        else:
            actions = self._sample_action(probs)

        bang_bang_actions = self._convert_to_bang_bang(actions)
        log_probs = self._compute_log_probs(probs, actions)

        return bang_bang_actions, log_probs

    def _get_deterministic_action(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs > 0.5).float()

    def _sample_action(self, probs: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Bernoulli(probs)
        return dist.sample()

    def _convert_to_bang_bang(self, actions: torch.Tensor) -> torch.Tensor:
        return 2.0 * actions - 1.0

    def _compute_log_probs(self, probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Bernoulli(probs)
        return dist.log_prob(actions).sum(dim=-1)