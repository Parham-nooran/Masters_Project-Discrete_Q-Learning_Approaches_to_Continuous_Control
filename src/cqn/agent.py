"""
Coarse-to-Fine Q-Network Agent for continuous control.
"""

from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.training_utils import huber_loss
from src.cqn.discretizer import CoarseToFineDiscretizer
from src.cqn.networks import CQNNetwork
from src.common.device_utils import (
    get_device,
    optimizer_step,
    mark_step,
    save_checkpoint as save_ckpt,
    load_checkpoint as load_ckpt
)


class CQNAgent:
    """
    Coarse-to-fine Q-Network agent for continuous control.

    Implements hierarchical action discretization where the agent iteratively
    refines action selection from coarse to fine levels.
    """

    def __init__(self, config, obs_shape: Tuple, action_spec: Dict):
        """
        Initialize CQN agent.

        Args:
            config: Configuration object with hyperparameters.
            obs_shape: Shape of observations.
            action_spec: Dictionary with 'low' and 'high' action bounds.
        """
        self.config = config
        self.obs_shape = obs_shape
        self.action_spec = action_spec
        self.device, self.is_tpu, self.use_amp = get_device()

        self._initialize_components()

    def _initialize_components(self) -> None:
        self.num_levels = self.config.num_levels
        self.num_bins = self.config.num_bins
        self.action_dim = len(self.action_spec["low"])

        self.network = CQNNetwork(
            self.config, self.obs_shape, self.action_dim, self.num_levels, self.num_bins
        ).to(self.device)

        self.discretizer = CoarseToFineDiscretizer(
            self.action_spec, self.num_levels, self.num_bins
        )

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.lr,
            weight_decay=1e-5
        )

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.replay_buffer_size,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            n_step=self.config.n_step,
            discount=self.config.discount,
        )
        self.replay_buffer.to_device(self.device)
        self._initialize_training_state()
        self._setup_amp()

    def _initialize_training_state(self) -> None:
        self.epsilon = self.config.initial_epsilon
        self.epsilon_decay = self.config.epsilon_decay
        self.min_epsilon = self.config.min_epsilon
        self.target_update_freq = self.config.target_update_freq
        self.training_steps = 0

    def _setup_amp(self) -> None:
        if self.use_amp and not self.is_tpu:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

    def select_action(self, obs: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """
        Select action using hierarchical coarse-to-fine strategy.
        Uses TARGET network for stability during evaluation and rollouts.
        """
        obs = self._prepare_observation(obs)

        with torch.no_grad():
            action = self._hierarchical_action_selection(obs, evaluate, use_target=True)

        action_low = torch.tensor(self.action_spec["low"], dtype=torch.float32, device=self.device)
        action_high = torch.tensor(self.action_spec["high"], dtype=torch.float32, device=self.device)
        action = torch.clamp(action, action_low, action_high)

        if not evaluate:
            noise = torch.randn_like(action) * 0.01
            action = torch.clamp(action + noise, action_low, action_high)

        return action.cpu()

    def _prepare_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == len(self.obs_shape):
            obs = obs.unsqueeze(0)

        return obs.to(self.device)

    def _hierarchical_action_selection(
        self, obs: torch.Tensor, evaluate: bool, use_target: bool = False
    ) -> torch.Tensor:
        """
        Perform hierarchical action selection across levels.

        KEY FIX: At each level, we discretize the REFINED range from the parent level,
        not the entire action space.
        """
        batch_size = obs.shape[0]
        device = obs.device

        action_low = torch.tensor(self.action_spec["low"], dtype=torch.float32, device=device)
        action_high = torch.tensor(self.action_spec["high"], dtype=torch.float32, device=device)

        range_min = action_low.unsqueeze(0).expand(batch_size, -1)
        range_max = action_high.unsqueeze(0).expand(batch_size, -1)

        prev_action = None

        for level in range(self.num_levels):
            q1, q2 = self.network(obs, level, prev_action, use_target=use_target)
            q_combined = torch.max(q1, q2)

            if not evaluate and torch.rand(1).item() < self.epsilon:
                level_actions = torch.randint(
                    0, self.num_bins, (batch_size, self.action_dim), device=device
                )
            else:
                level_actions = q_combined.argmax(dim=-1)

            bin_width = (range_max - range_min) / self.num_bins
            level_continuous = (range_min + (level_actions.float() + 0.5) * bin_width).float()

            if level < self.num_levels - 1:
                range_min = range_min + level_actions.float() * bin_width
                range_max = range_min + bin_width

                range_min = torch.clamp(range_min, action_low, action_high)
                range_max = torch.clamp(range_max, action_low, action_high)

            prev_action = level_continuous

        return prev_action[0]

    def get_current_bin_widths(self) -> Dict[str, float]:
        """
        Calculate current effective bin widths at finest level.
        With hierarchical refinement, the finest level has bin width = full_range / (num_bins^num_levels)
        """
        parent_range_width = (
            torch.tensor(self.action_spec["high"], dtype=torch.float32) -
            torch.tensor(self.action_spec["low"], dtype=torch.float32)
        )

        finest_level_factor = self.num_bins ** self.num_levels
        finest_bin_width = parent_range_width / finest_level_factor

        return {
            f"bin_width_dim_{i}": finest_bin_width[i].item()
            for i in range(self.action_dim)
        }

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch

        metrics = self._compute_and_apply_loss(
            obs, actions, rewards, next_obs, dones, discounts, weights
        )

        self._update_target_network()
        self._update_priorities(indices, metrics["td_errors"])
        self._update_epsilon()

        self.training_steps += 1

        return self._format_metrics(metrics)

    def _compute_and_apply_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict:
        if self.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                total_loss, td_errors, q1_last, q2_last = self._compute_loss(
                    obs, actions, rewards, next_obs, dones, discounts, weights
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()

            if hasattr(self.config, 'max_grad_norm'):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss, td_errors, q1_last, q2_last = self._compute_loss(
                obs, actions, rewards, next_obs, dones, discounts, weights
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if hasattr(self.config, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )

            optimizer_step(self.optimizer, self.is_tpu)
            mark_step(self.is_tpu)

        return {
            "loss": total_loss,
            "td_errors": td_errors,
            "q1_last": q1_last,
            "q2_last": q2_last,
        }

    def _compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, list, torch.Tensor, torch.Tensor]:
        """
        Compute hierarchical loss across all levels.

        KEY FIX: Use ONLINE network for next action selection (Double Q-Learning)
        and TARGET network for value estimation.
        """
        obs = obs.float()
        actions = actions.float()
        next_obs = next_obs.float()
        batch_size = obs.shape[0]
        device = obs.device

        total_loss = 0.0
        td_errors = []

        action_low = torch.tensor(self.action_spec["low"], dtype=torch.float32, device=device)
        action_high = torch.tensor(self.action_spec["high"], dtype=torch.float32, device=device)

        range_min_curr = action_low.unsqueeze(0).expand(batch_size, -1)
        range_max_curr = action_high.unsqueeze(0).expand(batch_size, -1)

        range_min_next = action_low.unsqueeze(0).expand(batch_size, -1)
        range_max_next = action_high.unsqueeze(0).expand(batch_size, -1)

        prev_action = None
        prev_next_action = None
        q1_last = None
        q2_last = None

        for level in range(self.num_levels):
            q1_current, q2_current = self.network(obs, level, prev_action, use_target=False)

            with torch.no_grad():
                q1_next_online, q2_next_online = self.network(
                    next_obs, level, prev_next_action, use_target=False
                )
                q_next_online = torch.max(q1_next_online, q2_next_online)
                next_actions = q_next_online.argmax(dim=2)

                q1_next_target, q2_next_target = self.network(
                    next_obs, level, prev_next_action, use_target=True
                )

                target_q1 = torch.gather(
                    q1_next_target, 2, next_actions.unsqueeze(2)
                ).squeeze(2)
                target_q2 = torch.gather(
                    q2_next_target, 2, next_actions.unsqueeze(2)
                ).squeeze(2)

                target_q = torch.min(target_q1, target_q2)

                td_target = (
                    rewards.unsqueeze(1)
                    + (1 - dones.float()).unsqueeze(1) * discounts.unsqueeze(1) * target_q
                )

            bin_width_curr = (range_max_curr - range_min_curr) / self.num_bins
            discrete_actions = torch.floor((actions - range_min_curr) / bin_width_curr).long()
            discrete_actions = torch.clamp(discrete_actions, 0, self.num_bins - 1)

            current_q1 = torch.gather(q1_current, 2, discrete_actions.unsqueeze(2)).squeeze(2)
            current_q2 = torch.gather(q2_current, 2, discrete_actions.unsqueeze(2)).squeeze(2)

            td_error1 = td_target - current_q1
            td_error2 = td_target - current_q2

            loss1 = huber_loss(td_error1, self.config.huber_loss_parameter)
            loss2 = huber_loss(td_error2, self.config.huber_loss_parameter)

            level_loss = ((loss1 + loss2) * weights.unsqueeze(1)).mean()

            td_errors_level = torch.abs(td_error1).mean(dim=1) + torch.abs(td_error2).mean(dim=1)

            total_loss += level_loss
            td_errors.append(td_errors_level)

            if level == self.num_levels - 1:
                q1_last = current_q1
                q2_last = current_q2

            if level < self.num_levels - 1:
                range_min_curr = range_min_curr + discrete_actions.float() * bin_width_curr
                range_max_curr = range_min_curr + bin_width_curr

                prev_action = (range_min_curr + 0.5 * bin_width_curr).float()

                with torch.no_grad():
                    bin_width_next = (range_max_next - range_min_next) / self.num_bins
                    range_min_next = range_min_next + next_actions.float() * bin_width_next
                    range_max_next = range_min_next + bin_width_next
                    prev_next_action = (range_min_next + 0.5 * bin_width_next).float()

        total_loss = total_loss / self.num_levels

        return total_loss, td_errors, q1_last, q2_last

    def _update_target_network(self) -> None:
        if self.training_steps % self.target_update_freq == 0:
            self.network.update_target_networks(tau=self.config.target_update_tau)

    def _update_priorities(self, indices: list, td_errors: list) -> None:
        avg_td_error = torch.stack(td_errors).mean(dim=0)
        self.replay_buffer.update_priorities(
            indices, avg_td_error.detach().cpu().numpy()
        )

    def _update_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _format_metrics(self, metrics: Dict) -> Dict[str, float]:
        return {
            "loss": metrics["loss"].item(),
            "epsilon": self.epsilon,
            "q_mean": (metrics["q1_last"].mean() + metrics["q2_last"].mean()).item() / 2,
            "mean_abs_td_error": torch.stack(metrics["td_errors"]).mean(dim=0).mean().item(),
        }

    def store_transition(
        self, obs, action: float, reward: float, next_obs, done: bool
    ) -> None:
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save_checkpoint(self, filepath: str, episode: int) -> None:
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "episode": episode,
            "config": self.config,
        }
        save_ckpt(checkpoint, filepath, self.is_tpu)

    def load_checkpoint(self, filepath: str) -> int:
        checkpoint = load_ckpt(filepath, self.device, self.is_tpu)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]
        return checkpoint.get("episode", 0)