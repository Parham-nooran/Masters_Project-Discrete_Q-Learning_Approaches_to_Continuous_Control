"""
Coarse-to-Fine Q-Network Agent for continuous control.
"""

from typing import Dict, Tuple

import numpy as np
import torch

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
        """Initialize network, buffer, and training utilities."""
        self.num_levels = self.config.num_levels
        self.num_bins = self.config.num_bins
        self.action_dim = len(self.action_spec["low"])

        self.network = CQNNetwork(
            self.config, self.obs_shape, self.action_dim, self.num_levels, self.num_bins
        ).to(self.device)

        self.discretizer = CoarseToFineDiscretizer(
            self.action_spec, self.num_levels, self.num_bins
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.lr)

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
        """Initialize training state variables."""
        self.epsilon = self.config.initial_epsilon
        self.epsilon_decay = self.config.epsilon_decay
        self.min_epsilon = self.config.min_epsilon
        self.target_update_freq = self.config.target_update_freq
        self.training_steps = 0

    def _setup_amp(self) -> None:
        """Setup automatic mixed precision based on device."""
        if self.use_amp and not self.is_tpu:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

    def select_action(self, obs: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """
        Select action using hierarchical coarse-to-fine strategy.

        Args:
            obs: Observation tensor
            evaluate: If True, use greedy action selection

        Returns:
            Selected continuous action
        """
        obs = self._prepare_observation(obs)

        with torch.no_grad():
            action = self._hierarchical_action_selection(obs, evaluate)

        action = action.cpu()
        action_low = torch.tensor(self.action_spec["low"], device=action.device)
        action_high = torch.tensor(self.action_spec["high"], device=action.device)
        action = torch.clamp(action, action_low, action_high)
        return action

    def _prepare_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Convert observation to proper format and device.

        Args:
            obs: Input observation

        Returns:
            Formatted observation tensor
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == len(self.obs_shape):
            obs = obs.unsqueeze(0)

        return obs.to(self.device)

    def _hierarchical_action_selection(
            self, obs: torch.Tensor, evaluate: bool
    ) -> torch.Tensor:
        """
        Hierarchical action selection from coarse to fine levels.

        Args:
            obs: Observation tensor [1, obs_dim]
            evaluate: If True, use greedy selection

        Returns:
            Final continuous action
        """
        prev_action = None
        level_continuous = None

        for level in range(self.num_levels):
            q1, q2 = self.network(obs, level, prev_action, use_target=False)
            q_combined = torch.max(q1, q2)

            if not evaluate and torch.rand(1).item() < self.epsilon:
                level_actions = torch.randint(
                    0, self.num_bins, (self.action_dim,), device=self.device
                )
            else:
                level_actions = q_combined[0].argmax(dim=-1)

            if level == 0:
                level_continuous = self.discretizer.discrete_to_continuous(
                    level_actions.unsqueeze(0), level, parent_action=None
                )[0]
            else:
                level_continuous = self.discretizer.discrete_to_continuous(
                    level_actions.unsqueeze(0), level, parent_action=prev_action
                )[0]

            if level < self.num_levels - 1:
                prev_action = level_continuous.unsqueeze(0)

        return level_continuous

    def get_current_bin_widths(self) -> Dict[str, float]:
        """
        Calculate current effective bin widths at finest level.

        Returns:
            Dictionary with bin width per action dimension
        """
        parent_range_width = torch.tensor(
            self.action_spec["high"], device=self.device
        ) - torch.tensor(self.action_spec["low"], device=self.device)

        finest_level_factor = self.num_bins ** (self.num_levels - 1)
        finest_bin_width = parent_range_width / finest_level_factor / self.num_bins

        return {
            f"bin_width_dim_{i}": finest_bin_width[i].item()
            for i in range(self.action_dim)
        }

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Update networks using batch from replay buffer.

        Args:
            batch_size: Number of samples per batch.

        Returns:
            Dictionary of training metrics.
        """
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
        """
        Compute loss and apply backpropagation.

        Args:
            obs: Observation batch
            actions: Action batch
            rewards: Reward batch
            next_obs: Next observation batch
            dones: Done flags batch
            discounts: Discount factors batch
            weights: Priority weights batch

        Returns:
            Dictionary with loss and metrics
        """
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

    def _continuous_to_discrete_vectorized(self, continuous_actions: torch.Tensor) -> torch.Tensor:
        """
        Vectorized conversion of continuous actions to discrete indices for ALL levels.

        Args:
            continuous_actions: [batch, action_dim]

        Returns:
            discrete_actions: [batch, num_levels, action_dim]
        """
        batch_size = continuous_actions.shape[0]
        discrete_all = torch.zeros(batch_size, self.num_levels, self.action_dim, dtype=torch.long, device=self.device)

        bins_level0 = torch.stack([self.discretizer.action_bins[0][dim] for dim in range(self.action_dim)], dim=0)
        distance_0 = torch.abs(continuous_actions.unsqueeze(2) - bins_level0.unsqueeze(0))
        discrete_all[:, 0, :] = distance_0.argmin(dim=2)

        for level in range(1, self.num_levels):
            parent_actions = self.discretizer.discrete_to_continuous(
                discrete_all[:, level - 1, :],
                level - 1,
                parent_action=None
            )

            action_ranges = self.discretizer.get_action_range_for_level(level, parent_actions)
            range_min, range_max = action_ranges[:, 0, :], action_ranges[:, 1, :]

            bin_width = (range_max - range_min) / self.num_bins
            bins_for_level = range_min.unsqueeze(2) + bin_width.unsqueeze(2) * torch.arange(
                self.num_bins, device=self.device
            ).view(1, 1, self.num_bins)

            distance = torch.abs(continuous_actions.unsqueeze(2) - bins_for_level)
            discrete_all[:, level, :] = distance.argmin(dim=2)

        return discrete_all

    def _compute_prev_actions_vectorized(self, discrete_actions_all: torch.Tensor) -> torch.Tensor:
        """
        Compute previous level actions for all levels in a vectorized manner.

        Args:
            discrete_actions_all: [batch, num_levels, action_dim]

        Returns:
            prev_actions: [batch, num_levels, action_dim]
        """
        batch_size = discrete_actions_all.shape[0]
        prev_actions = torch.zeros(batch_size, self.num_levels, self.action_dim, dtype=torch.float32, device=self.device)

        prev_actions[:, 0, :] = 0.0

        for level in range(1, self.num_levels):
            parent_action_input = prev_actions[:, level - 1, :] if level > 1 else None
            prev_actions[:, level, :] = self.discretizer.discrete_to_continuous(
                discrete_actions_all[:, level - 1, :],
                level - 1,
                parent_action=parent_action_input
            )

        return prev_actions

    def _compute_prev_actions_vectorized_online(
        self, next_obs: torch.Tensor, discrete_actions_current: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute previous actions for next states using current discrete actions.

        Args:
            next_obs: [batch, obs_dim]
            discrete_actions_current: [batch, num_levels, action_dim]

        Returns:
            prev_next_actions: [batch, num_levels, action_dim]
        """
        batch_size = next_obs.shape[0]
        prev_next_actions = torch.zeros(
            batch_size, self.num_levels, self.action_dim, dtype=torch.float32, device=self.device
        )

        prev_next_actions[:, 0, :] = 0.0

        for level in range(1, self.num_levels):
            parent_input = prev_next_actions[:, level - 1, :] if level > 1 else None
            prev_next_actions[:, level, :] = self.discretizer.discrete_to_continuous(
                discrete_actions_current[:, level - 1, :],
                level - 1,
                parent_action=parent_input
            )

        return prev_next_actions

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
        Fully vectorized hierarchical loss computation across all levels simultaneously.

        Args:
            obs: [batch, obs_dim]
            actions: [batch, action_dim]
            rewards: [batch]
            next_obs: [batch, obs_dim]
            dones: [batch]
            discounts: [batch]
            weights: [batch]

        Returns:
            Tuple of (total_loss, td_errors_list, q1_last, q2_last)
        """
        batch_size = obs.shape[0]

        discrete_actions_all = self._continuous_to_discrete_vectorized(actions)

        obs_expanded = obs.unsqueeze(1).expand(batch_size, self.num_levels, -1).reshape(
            batch_size * self.num_levels, -1
        )
        next_obs_expanded = next_obs.unsqueeze(1).expand(batch_size, self.num_levels, -1).reshape(
            batch_size * self.num_levels, -1
        )

        prev_actions_all = self._compute_prev_actions_vectorized(discrete_actions_all)
        prev_actions_expanded = prev_actions_all.reshape(batch_size * self.num_levels, self.action_dim)

        level_indices = torch.arange(self.num_levels, device=self.device).unsqueeze(0).expand(
            batch_size, -1
        ).reshape(-1)

        q1_current_all, q2_current_all = self.network.forward_batched_levels(
            obs_expanded, level_indices, prev_actions_expanded, use_target=False
        )

        with torch.no_grad():
            prev_next_actions_all = self._compute_prev_actions_vectorized_online(next_obs, discrete_actions_all)
            prev_next_actions_expanded = prev_next_actions_all.reshape(batch_size * self.num_levels, self.action_dim)

            q1_next_online, q2_next_online = self.network.forward_batched_levels(
                next_obs_expanded, level_indices, prev_next_actions_expanded, use_target=False
            )
            q_next_online = torch.max(q1_next_online, q2_next_online)
            next_actions = q_next_online.argmax(dim=2)

            q1_next_target, q2_next_target = self.network.forward_batched_levels(
                next_obs_expanded, level_indices, prev_next_actions_expanded, use_target=True
            )

            target_q1 = torch.gather(q1_next_target, 2, next_actions.unsqueeze(2)).squeeze(2)
            target_q2 = torch.gather(q2_next_target, 2, next_actions.unsqueeze(2)).squeeze(2)
            target_q = torch.min(target_q1, target_q2)

            target_q = target_q.view(batch_size, self.num_levels, self.action_dim)
            td_target = (
                rewards.unsqueeze(1).unsqueeze(2) +
                (1 - dones.float()).unsqueeze(1).unsqueeze(2) *
                discounts.unsqueeze(1).unsqueeze(2) * target_q
            )
            td_target = td_target.view(batch_size * self.num_levels, self.action_dim)

        discrete_actions_expanded = discrete_actions_all.reshape(batch_size * self.num_levels, self.action_dim)
        current_q1 = torch.gather(q1_current_all, 2, discrete_actions_expanded.unsqueeze(2)).squeeze(2)
        current_q2 = torch.gather(q2_current_all, 2, discrete_actions_expanded.unsqueeze(2)).squeeze(2)

        td_error1 = td_target - current_q1
        td_error2 = td_target - current_q2

        loss1 = huber_loss(td_error1, self.config.huber_loss_parameter)
        loss2 = huber_loss(td_error2, self.config.huber_loss_parameter)

        weights_expanded = weights.unsqueeze(1).unsqueeze(2).expand(
            batch_size, self.num_levels, self.action_dim
        ).reshape(batch_size * self.num_levels, self.action_dim)

        total_loss = ((loss1 + loss2) * weights_expanded).sum() / (batch_size * self.num_levels)

        td_errors = (torch.abs(td_error1) + torch.abs(td_error2)).view(
            batch_size, self.num_levels, self.action_dim
        ).mean(dim=2)
        td_errors_list = [td_errors[:, i] for i in range(self.num_levels)]

        q1_last = current_q1.view(batch_size, self.num_levels, self.action_dim)[:, -1, :]
        q2_last = current_q2.view(batch_size, self.num_levels, self.action_dim)[:, -1, :]

        return total_loss, td_errors_list, q1_last, q2_last

    def _update_target_network(self) -> None:
        """Update target networks with soft updates."""
        if self.training_steps % self.target_update_freq == 0:
            self.network.update_target_networks(tau=self.config.target_update_tau)

    def _update_priorities(self, indices: list, td_errors: list) -> None:
        """
        Update replay buffer priorities based on TD errors.

        Args:
            indices: Batch indices
            td_errors: TD error tensors from each level
        """
        avg_td_error = torch.stack(td_errors).mean(dim=0)
        self.replay_buffer.update_priorities(
            indices, avg_td_error.detach().cpu().numpy()
        )

    def _update_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _format_metrics(self, metrics: Dict) -> Dict[str, float]:
        """
        Format metrics for logging.

        Args:
            metrics: Raw metrics dictionary

        Returns:
            Formatted metrics dictionary
        """
        return {
            "loss": metrics["loss"].item(),
            "epsilon": self.epsilon,
            "q_mean": (metrics["q1_last"].mean() + metrics["q2_last"].mean()).item() / 2,
            "mean_abs_td_error": torch.stack(metrics["td_errors"]).mean(dim=0).mean().item(),
        }

    def store_transition(
        self, obs, action: float, reward: float, next_obs, done: bool
    ) -> None:
        """
        Store transition in replay buffer.

        Args:
            obs: Observation
            action: Continuous action
            reward: Reward value
            next_obs: Next observation
            done: Episode termination flag
        """
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save(self, filepath: str) -> None:
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "config": self.config,
        }
        save_ckpt(checkpoint, filepath, self.is_tpu)

    def save_checkpoint(self, filepath: str, episode: int) -> None:
        """
        Save agent checkpoint with episode information.

        Args:
            filepath: Path to save checkpoint
            episode: Current episode number
        """
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
        """
        Load agent checkpoint and return episode number.

        Args:
            filepath: Path to checkpoint

        Returns:
            Episode number from checkpoint
        """
        checkpoint = load_ckpt(filepath, self.device, self.is_tpu)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]
        return checkpoint.get("episode", 0)

    def load(self, filepath: str) -> None:
        """
        Load agent checkpoint.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = load_ckpt(filepath, self.device, self.is_tpu)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]