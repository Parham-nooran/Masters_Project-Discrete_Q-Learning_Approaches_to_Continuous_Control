from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.common.replay_buffer import PrioritizedReplayBuffer
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
    def __init__(self, config, obs_shape: Tuple, action_spec: Dict):
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
            weight_decay=self.config.weight_decay
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
        obs = self._prepare_observation(obs)

        with torch.no_grad():
            action = self._hierarchical_action_selection(obs, evaluate)

        action = action.cpu()
        action_low = torch.tensor(self.action_spec["low"], device=action.device)
        action_high = torch.tensor(self.action_spec["high"], device=action.device)
        action = torch.clamp(action, action_low, action_high)
        return action

    def _prepare_observation(self, obs: torch.Tensor) -> torch.Tensor:
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
        prev_action = None
        level_continuous = None

        for level in range(self.num_levels):
            q1_dist, q2_dist = self.network(obs, level, prev_action, use_target=True)

            q1_values = self.network.get_q_values(q1_dist)
            q2_values = self.network.get_q_values(q2_dist)
            q_combined = torch.max(q1_values, q2_values)

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
                total_loss, td_errors, mean_q = self._compute_c51_loss(
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
            total_loss, td_errors, mean_q = self._compute_c51_loss(
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
            "mean_q": mean_q,
        }

    def _continuous_to_discrete_vectorized(self, continuous_actions: torch.Tensor) -> torch.Tensor:
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

            action_ranges = self.discretizer.get_action_range_for_level_batch(level, parent_actions)
            range_min, range_max = action_ranges[:, 0, :], action_ranges[:, 1, :]

            bin_width = (range_max - range_min) / self.num_bins
            bins_for_level = range_min.unsqueeze(2) + bin_width.unsqueeze(2) * torch.arange(
                self.num_bins, device=self.device
            ).view(1, 1, self.num_bins)

            distance = torch.abs(continuous_actions.unsqueeze(2) - bins_for_level)
            discrete_all[:, level, :] = distance.argmin(dim=2)

        return discrete_all

    def _compute_prev_actions_vectorized(self, discrete_actions_all: torch.Tensor) -> torch.Tensor:
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

    def _compute_c51_loss(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_obs: torch.Tensor,
            dones: torch.Tensor,
            discounts: torch.Tensor,
            weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:
        obs = obs.float()
        actions = actions.float()
        next_obs = next_obs.float()
        rewards = rewards.float()
        dones = dones.float()
        discounts = discounts.float()
        weights = weights.float()

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

        q1_dist_current, q2_dist_current = self.network.forward_batched_levels(
            obs_expanded, level_indices, prev_actions_expanded, use_target=False
        )

        with torch.no_grad():
            prev_next_actions_all = self._compute_prev_actions_vectorized(discrete_actions_all)
            prev_next_actions_expanded = prev_next_actions_all.reshape(batch_size * self.num_levels, self.action_dim)

            q1_dist_next_online, q2_dist_next_online = self.network.forward_batched_levels(
                next_obs_expanded, level_indices, prev_next_actions_expanded, use_target=False
            )

            q1_next_online = self.network.get_q_values(q1_dist_next_online)
            q2_next_online = self.network.get_q_values(q2_dist_next_online)
            q_next_online = torch.max(q1_next_online, q2_next_online)
            next_actions = q_next_online.argmax(dim=2)

            q1_dist_next_target, q2_dist_next_target = self.network.forward_batched_levels(
                next_obs_expanded, level_indices, prev_next_actions_expanded, use_target=True
            )

            batch_idx = torch.arange(batch_size * self.num_levels, device=self.device)
            dim_idx = torch.arange(self.action_dim, device=self.device).unsqueeze(0).expand(
                batch_size * self.num_levels, -1
            )

            q1_dist_selected = q1_dist_next_target[batch_idx.unsqueeze(1), dim_idx, next_actions]
            q2_dist_selected = q2_dist_next_target[batch_idx.unsqueeze(1), dim_idx, next_actions]

            q1_values_selected = self.network.get_q_values(q1_dist_selected.unsqueeze(2)).squeeze(2)
            q2_values_selected = self.network.get_q_values(q2_dist_selected.unsqueeze(2)).squeeze(2)

            target_dist_selector = torch.where(
                q1_values_selected > q2_values_selected,
                q1_dist_selected,
                q2_dist_selected
            )

            target_dist_selector = target_dist_selector.view(batch_size, self.num_levels, self.action_dim, self.network.num_atoms)

            rewards_expanded = rewards.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
                batch_size, self.num_levels, self.action_dim, self.network.num_atoms
            )
            dones_expanded = dones.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
                batch_size, self.num_levels, self.action_dim, self.network.num_atoms
            )
            discounts_expanded = discounts.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
                batch_size, self.num_levels, self.action_dim, self.network.num_atoms
            )

            support = self.network.support.to(self.device)

            tz = rewards_expanded + (1 - dones_expanded) * discounts_expanded * support.view(1, 1, 1, -1)
            tz = torch.clamp(tz, self.network.v_min, self.network.v_max)

            b = (tz - self.network.v_min) / self.network.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            l = torch.clamp(l, 0, self.network.num_atoms - 1)
            u = torch.clamp(u, 0, self.network.num_atoms - 1)

            target_dist = torch.zeros_like(target_dist_selector)

            offset = torch.arange(batch_size * self.num_levels * self.action_dim, device=self.device) * self.network.num_atoms
            offset = offset.view(batch_size, self.num_levels, self.action_dim, 1).expand_as(l)

            target_dist_flat = target_dist.view(-1)
            target_dist_selector_flat = target_dist_selector.view(batch_size, self.num_levels, self.action_dim, -1)

            l_flat = (l + offset).view(-1)
            u_flat = (u + offset).view(-1)

            m_l = target_dist_selector_flat * (u.float() - b)
            m_u = target_dist_selector_flat * (b - l.float())

            target_dist_flat.index_add_(0, l_flat, m_l.view(-1))
            target_dist_flat.index_add_(0, u_flat, m_u.view(-1))

            target_dist = target_dist_flat.view(batch_size, self.num_levels, self.action_dim, self.network.num_atoms)
            target_dist = target_dist.view(batch_size * self.num_levels, self.action_dim, self.network.num_atoms)

        discrete_actions_expanded = discrete_actions_all.reshape(batch_size * self.num_levels, self.action_dim)

        current_dist_selected = q1_dist_current[batch_idx.unsqueeze(1), dim_idx, discrete_actions_expanded]
        current_dist2_selected = q2_dist_current[batch_idx.unsqueeze(1), dim_idx, discrete_actions_expanded]

        log_p1 = torch.log(current_dist_selected + 1e-8)
        log_p2 = torch.log(current_dist2_selected + 1e-8)

        loss1 = -(target_dist * log_p1).sum(dim=-1)
        loss2 = -(target_dist * log_p2).sum(dim=-1)

        weights_expanded = weights.unsqueeze(1).unsqueeze(2).expand(
            batch_size, self.num_levels, self.action_dim
        ).reshape(batch_size * self.num_levels, self.action_dim)

        weighted_loss = (loss1 + loss2) * weights_expanded
        total_loss = weighted_loss.sum() / (batch_size * self.num_levels * self.action_dim)

        with torch.no_grad():
            current_q1 = self.network.get_q_values(current_dist_selected.unsqueeze(2)).squeeze(2)
            current_q2 = self.network.get_q_values(current_dist2_selected.unsqueeze(2)).squeeze(2)

            target_q = self.network.get_q_values(target_dist.unsqueeze(2)).squeeze(2)

            td_error1 = torch.abs(target_q - current_q1)
            td_error2 = torch.abs(target_q - current_q2)
            td_errors_combined = (td_error1 + td_error2).view(batch_size, self.num_levels, self.action_dim)
            td_errors_list = [td_errors_combined[:, i, :].mean(dim=1) for i in range(self.num_levels)]

            mean_q = (current_q1.mean() + current_q2.mean()) / 2

        return total_loss, td_errors_list, mean_q

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
            "q_mean": metrics["mean_q"].item(),
            "mean_abs_td_error": torch.stack(metrics["td_errors"]).mean().item(),
        }

    def store_transition(
        self, obs, action: float, reward: float, next_obs, done: bool
    ) -> None:
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save(self, filepath: str) -> None:
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "config": self.config,
        }
        save_ckpt(checkpoint, filepath, self.is_tpu)

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

    def load(self, filepath: str) -> None:
        checkpoint = load_ckpt(filepath, self.device, self.is_tpu)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]