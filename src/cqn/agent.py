from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.common.device_utils import (
    get_device,
    optimizer_step,
    mark_step,
    save_checkpoint as save_ckpt,
    load_checkpoint as load_ckpt
)
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.cqn.discretizer import CoarseToFineDiscretizer
from src.cqn.networks import CQNNetwork


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
            weight_decay=0.1
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
        batch_size = obs.shape[0]
        device = obs.device

        action_low = torch.tensor(self.action_spec["low"], dtype=torch.float32, device=device)
        action_high = torch.tensor(self.action_spec["high"], dtype=torch.float32, device=device)

        low = action_low.unsqueeze(0).expand(batch_size, -1)
        high = action_high.unsqueeze(0).expand(batch_size, -1)

        prev_action = torch.zeros(batch_size, self.action_dim, device=device)

        for level in range(self.num_levels):
            q_logits = self.network(obs, level, prev_action, use_target=use_target)

            q_probs = F.softmax(q_logits, dim=-1)
            qs = (q_probs * self.network.support.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(-1)

            if not evaluate and torch.rand(1).item() < self.epsilon:
                level_actions = torch.randint(
                    0, self.num_bins, (batch_size, self.action_dim), device=device
                )
            else:
                level_actions = qs.argmax(dim=-1)

            bin_width = (high - low) / self.num_bins
            level_continuous = low + (level_actions.float() + 0.5) * bin_width

            if level < self.num_levels - 1:
                low = low + level_actions.float() * bin_width
                high = low + bin_width
                low = torch.clamp(low, action_low, action_high)
                high = torch.clamp(high, action_low, action_high)

            prev_action = (low + high) / 2.0

        return prev_action[0]

    def get_current_bin_widths(self) -> Dict[str, float]:
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
                total_loss, td_errors = self._compute_loss(
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
            total_loss, td_errors = self._compute_loss(
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
    ) -> Tuple[torch.Tensor, list]:
        obs = obs.float()
        actions = actions.float()
        next_obs = next_obs.float()
        batch_size = obs.shape[0]
        device = obs.device

        action_low = torch.tensor(self.action_spec["low"], dtype=torch.float32, device=device)
        action_high = torch.tensor(self.action_spec["high"], dtype=torch.float32, device=device)

        discrete_actions = self._encode_action(actions, action_low, action_high)

        with torch.no_grad():
            next_action, _ = self._get_next_action(next_obs, action_low, action_high)
            next_discrete_actions = self._encode_action(next_action, action_low, action_high)

        low = action_low.unsqueeze(0).expand(batch_size, -1)
        high = action_high.unsqueeze(0).expand(batch_size, -1)

        low_next = action_low.unsqueeze(0).expand(batch_size, -1)
        high_next = action_high.unsqueeze(0).expand(batch_size, -1)

        prev_action = torch.zeros(batch_size, self.action_dim, device=device)
        prev_next_action = torch.zeros(batch_size, self.action_dim, device=device)

        total_loss = 0.0
        td_errors = []

        for level in range(self.num_levels):
            q_logits_current = self.network(obs, level, prev_action, use_target=False)

            with torch.no_grad():
                q_logits_next = self.network(next_obs, level, prev_next_action, use_target=True)

                next_q_probs = F.softmax(q_logits_next, dim=-1)

                target_dist = self._compute_target_distribution(
                    next_q_probs, next_discrete_actions[:, level, :], rewards, discounts, dones
                )

            current_action_indices = discrete_actions[:, level, :].long()

            log_q_probs = F.log_softmax(q_logits_current, dim=-1)
            log_q_probs_a = torch.gather(
                log_q_probs,
                dim=-2,
                index=current_action_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.network.num_atoms)
            ).squeeze(-2)

            loss = -(target_dist * log_q_probs_a).sum(-1).mean(-1)

            weighted_loss = (loss * weights).mean()
            total_loss += weighted_loss

            td_errors.append(loss.detach())

            if level < self.num_levels - 1:
                bin_width = (high - low) / self.num_bins
                low = low + current_action_indices.float() * bin_width
                high = low + bin_width
                prev_action = (low + high) / 2.0

                bin_width_next = (high_next - low_next) / self.num_bins
                low_next = low_next + next_discrete_actions[:, level, :].float() * bin_width_next
                high_next = low_next + bin_width_next
                prev_next_action = (low_next + high_next) / 2.0

        total_loss = total_loss / self.num_levels

        return total_loss, td_errors

    def _encode_action(self, continuous_action, action_low, action_high):
        batch_size = continuous_action.shape[0]
        device = continuous_action.device

        low = action_low.unsqueeze(0).expand(batch_size, -1)
        high = action_high.unsqueeze(0).expand(batch_size, -1)

        discrete_actions = []

        for level in range(self.num_levels):
            bin_width = (high - low) / self.num_bins
            idx = torch.floor((continuous_action - low) / bin_width)
            idx = torch.clamp(idx, 0, self.num_bins - 1)
            discrete_actions.append(idx)

            recalculated_action = low + bin_width * idx
            low = recalculated_action
            high = recalculated_action + bin_width
            low = torch.clamp(low, action_low, action_high)
            high = torch.clamp(high, action_low, action_high)

        return torch.stack(discrete_actions, dim=1)

    def _get_next_action(self, next_obs, action_low, action_high):
        batch_size = next_obs.shape[0]
        device = next_obs.device

        low = action_low.unsqueeze(0).expand(batch_size, -1)
        high = action_high.unsqueeze(0).expand(batch_size, -1)
        prev_action = torch.zeros(batch_size, self.action_dim, device=device)

        for level in range(self.num_levels):
            q_logits = self.network(next_obs, level, prev_action, use_target=False)
            q_probs = F.softmax(q_logits, dim=-1)
            qs = (q_probs * self.network.support.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(-1)

            argmax_q = qs.argmax(dim=-1)

            bin_width = (high - low) / self.num_bins

            if level < self.num_levels - 1:
                low = low + argmax_q.float() * bin_width
                high = low + bin_width
                prev_action = (low + high) / 2.0

        continuous_action = (high + low) / 2.0
        return continuous_action, {}

    def _compute_target_distribution(self, next_q_probs, next_actions, rewards, discounts, dones):
        batch_size = next_q_probs.shape[0]

        next_q_probs_a = torch.gather(
            next_q_probs,
            dim=-2,
            index=next_actions.long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.network.num_atoms)
        ).squeeze(-2)

        shape = next_q_probs_a.shape
        next_q_probs_a = next_q_probs_a.view(-1, self.network.num_atoms)
        batch_size_flat = next_q_probs_a.shape[0]

        Tz = rewards.unsqueeze(1) + (1 - dones.float()).unsqueeze(1) * discounts.unsqueeze(
            1) * self.network.support.unsqueeze(0)
        Tz = Tz.clamp(min=self.network.v_min, max=self.network.v_max)

        b = (Tz - self.network.v_min) / self.network.delta_z
        lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)

        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (self.network.num_atoms - 1)) * (lower == upper)] += 1

        multiplier = batch_size_flat // lower.shape[0]
        b = torch.repeat_interleave(b, multiplier, 0)
        lower = torch.repeat_interleave(lower, multiplier, 0)
        upper = torch.repeat_interleave(upper, multiplier, 0)

        m = torch.zeros_like(next_q_probs_a)
        offset = (
            torch.linspace(
                0,
                ((batch_size_flat - 1) * self.network.num_atoms),
                batch_size_flat,
                device=lower.device,
                dtype=lower.dtype,
            )
            .unsqueeze(1)
            .expand(batch_size_flat, self.network.num_atoms)
        )

        m.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_q_probs_a * (upper.float() - b)).view(-1),
        )
        m.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_q_probs_a * (b - lower.float())).view(-1),
        )

        m = m.view(*shape)
        return m

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
