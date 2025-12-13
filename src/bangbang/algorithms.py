from abc import ABC, abstractmethod
import torch
from typing import Tuple, Optional, Dict
from src.common.networks import LayerNormMLP
import torch.optim as optim


class Base(ABC):
    @abstractmethod
    def compute_loss(
        self,
        policy,
        value_function,
        obs_encoded: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs_encoded: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        pass

    def update_target_networks(self):
        pass

    def get_optimizers(self):
        return []


def _compute_advantages(
    value_function, values, rewards, next_obs_encoded, dones, discounts
):
    with torch.no_grad():
        next_values = value_function(next_obs_encoded).squeeze(-1)
        td_targets = rewards + discounts * next_values * (~dones)
        advantages = td_targets - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def _compute_log_probs(policy, obs_encoded, actions):
    """Compute log probabilities for binary actions (0 or 1)."""
    logits = policy(obs_encoded)
    probs = torch.sigmoid(logits)
    dist = torch.distributions.Bernoulli(probs)
    return dist.log_prob(actions).sum(dim=-1)


def _compute_value_loss(
    values, rewards, next_obs_encoded, dones, discounts, value_function
):
    with torch.no_grad():
        next_values = value_function(next_obs_encoded).squeeze(-1)
        td_targets = rewards + discounts * next_values * (~dones)
    return torch.nn.functional.mse_loss(values, td_targets)


def _compute_td_errors(
    values, rewards, next_obs_encoded, dones, discounts, value_function
):
    with torch.no_grad():
        next_values = value_function(next_obs_encoded).squeeze(-1)
        td_targets = rewards + discounts * next_values * (~dones)
        td_errors = torch.abs(td_targets - values)
    return td_errors


class PPO(Base):

    def __init__(self, clip_ratio: float = 0.2, value_coef: float = 0.5):
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.old_log_probs = None

    def compute_loss(
        self,
        policy,
        value_function,
        obs_encoded: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs_encoded: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        values = value_function(obs_encoded).squeeze(-1)

        advantages = _compute_advantages(
            value_function, values, rewards, next_obs_encoded, dones, discounts
        )
        log_probs = _compute_log_probs(policy, obs_encoded, actions)

        with torch.no_grad():
            old_log_probs = log_probs.detach()

        policy_loss = self._compute_policy_loss(log_probs, old_log_probs, advantages)
        value_loss = _compute_value_loss(
            values, rewards, next_obs_encoded, dones, discounts, value_function
        )

        total_loss = policy_loss + self.value_coef * value_loss

        self.old_log_probs = log_probs.detach()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_advantage": advantages.mean().item(),
            "mean_value": values.mean().item(),
        }

        td_errors = _compute_td_errors(
            values, rewards, next_obs_encoded, dones, discounts, value_function
        )
        return total_loss, metrics, td_errors

    def _compute_policy_loss(self, log_probs, old_log_probs, advantages):
        ratio = torch.exp(log_probs - old_log_probs)
        ratio = torch.clamp(ratio, 0.0, 10.0)

        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        return -torch.min(ratio * advantages, clipped_ratio * advantages).mean()


class SAC(Base):

    def __init__(
        self, alpha: float = 0.2, tau: float = 0.005, learning_rate: float = 3e-4
    ):
        self.alpha = alpha
        self.tau = tau
        self.learning_rate = learning_rate
        self.target_q1 = None
        self.target_q2 = None
        self.q1 = None
        self.q2 = None
        self.q1_optimizer = None
        self.q2_optimizer = None

    def initialize_q_networks(self, input_size: int, device: torch.device):
        self.q1 = LayerNormMLP([input_size, 512, 512, 1], activate_final=False).to(
            device
        )
        self.q2 = LayerNormMLP([input_size, 512, 512, 1], activate_final=False).to(
            device
        )

        self.target_q1 = LayerNormMLP(
            [input_size, 512, 512, 1], activate_final=False
        ).to(device)
        self.target_q2 = LayerNormMLP(
            [input_size, 512, 512, 1], activate_final=False
        ).to(device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.learning_rate)

    def compute_loss(
        self,
        policy,
        value_function,
        obs_encoded: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs_encoded: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        q_loss, q1_values, q_target = self._compute_q_loss(
            obs_encoded, actions, rewards, next_obs_encoded, dones, discounts, policy
        )

        policy_loss = self._compute_policy_loss(policy, obs_encoded)

        total_loss = q_loss + policy_loss

        metrics = {
            "policy_loss": policy_loss.item(),
            "q1_loss": (q_loss / 2).item(),
            "q2_loss": (q_loss / 2).item(),
            "mean_q": q1_values.mean().item(),
        }

        td_errors = torch.abs(q_target - q1_values).detach()
        return total_loss, metrics, td_errors

    def _compute_q_loss(
        self, obs_encoded, actions, rewards, next_obs_encoded, dones, discounts, policy
    ):
        bang_bang_actions = 2.0 * actions - 1.0
        obs_action = torch.cat([obs_encoded, bang_bang_actions], dim=-1)
        q1_values = self.q1(obs_action).squeeze(-1)
        q2_values = self.q2(obs_action).squeeze(-1)

        with torch.no_grad():
            q_target = self._compute_q_target(
                policy, next_obs_encoded, rewards, dones, discounts
            )

        q1_loss = torch.nn.functional.mse_loss(q1_values, q_target)
        q2_loss = torch.nn.functional.mse_loss(q2_values, q_target)

        return q1_loss + q2_loss, q1_values, q_target

    def _compute_q_target(self, policy, next_obs_encoded, rewards, dones, discounts):
        logits_next = policy(next_obs_encoded)
        probs_next = torch.sigmoid(logits_next)
        dist_next = torch.distributions.Bernoulli(probs_next)
        next_actions_binary = dist_next.sample()
        log_probs_next = dist_next.log_prob(next_actions_binary).sum(dim=-1)
        next_actions_bangbang = 2.0 * next_actions_binary - 1.0
        next_obs_action = torch.cat([next_obs_encoded, next_actions_bangbang], dim=-1)
        target_q1 = self.target_q1(next_obs_action).squeeze(-1)
        target_q2 = self.target_q2(next_obs_action).squeeze(-1)
        target_q = torch.min(target_q1, target_q2) - self.alpha * log_probs_next

        return rewards + discounts * target_q * (1 - dones.float())

    def _compute_policy_loss(self, policy, obs_encoded):
        logits = policy(obs_encoded)
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs)
        new_actions_binary = dist.sample()
        log_probs = dist.log_prob(new_actions_binary).sum(dim=-1)
        new_actions_bangbang = 2.0 * new_actions_binary - 1.0
        new_obs_action = torch.cat([obs_encoded, new_actions_bangbang], dim=-1)
        q1_new = self.q1(new_obs_action).squeeze(-1)
        q2_new = self.q2(new_obs_action).squeeze(-1)
        q_new = torch.min(q1_new, q2_new)

        return (self.alpha * log_probs - q_new).mean()

    def update_target_networks(self):
        self._soft_update_target_network(self.target_q1, self.q1)
        self._soft_update_target_network(self.target_q2, self.q2)

    def _soft_update_target_network(self, target_network, source_network):
        for target_param, param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def get_optimizers(self):
        return [self.q1_optimizer, self.q2_optimizer]


def _compute_kl_divergence(policy, obs_encoded):
    with torch.no_grad():
        old_logits = policy(obs_encoded).detach()
        old_probs = torch.sigmoid(old_logits)

    logits = policy(obs_encoded)
    probs = torch.sigmoid(logits)

    kl_div = torch.sum(
        old_probs * torch.log(old_probs / (probs + 1e-8) + 1e-8)
        + (1 - old_probs) * torch.log((1 - old_probs) / (1 - probs + 1e-8) + 1e-8),
        dim=-1,
    ).mean()
    return kl_div


class MPO(Base):

    def __init__(self, epsilon: float = 0.1, epsilon_penalty: float = 0.001):
        self.epsilon = epsilon
        self.epsilon_penalty = epsilon_penalty
        self.log_temperature = torch.tensor(0.0)
        self.log_alpha_mean = torch.tensor(0.0)

    def compute_loss(
        self,
        policy,
        value_function,
        obs_encoded: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs_encoded: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        values = value_function(obs_encoded).squeeze(-1)

        advantages = _compute_advantages(
            value_function, values, rewards, next_obs_encoded, dones, discounts
        )
        normalized_weights = self._compute_normalized_weights(advantages)
        log_probs = _compute_log_probs(policy, obs_encoded, actions)
        policy_loss = -(normalized_weights * log_probs).sum()

        kl_div = _compute_kl_divergence(policy, obs_encoded)
        value_loss = _compute_value_loss(
            values, rewards, next_obs_encoded, dones, discounts, value_function
        )

        alpha_mean = torch.exp(self.log_alpha_mean)
        total_loss = policy_loss + value_loss + alpha_mean * (self.epsilon - kl_div)

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_divergence": kl_div.item(),
            "temperature": torch.exp(self.log_temperature).item(),
        }

        td_errors = _compute_td_errors(
            values, rewards, next_obs_encoded, dones, discounts, value_function
        )
        return total_loss, metrics, td_errors

    def _compute_normalized_weights(self, advantages):
        temperature = torch.exp(self.log_temperature)
        return torch.nn.functional.softmax(advantages / temperature, dim=0)
